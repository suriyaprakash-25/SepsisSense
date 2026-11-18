import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type {
  AdminUsersResponse,
  AssessFieldsResponse,
  AssessRequest,
  AssessResponse,
  DemoAccount,
  Meta,
  PatientDetail,
  ServerAck,
  SessionUser,
  WardResponse,
} from "./types";

export class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
    public errors: string[] = [message],
  ) {
    super(message);
  }
}

async function parseError(res: Response): Promise<HttpError> {
  let errors = [`${res.status} ${res.statusText}`];
  try {
    const j = await res.json();
    if (Array.isArray(j?.detail?.errors)) errors = j.detail.errors;
    else if (Array.isArray(j?.detail))
      errors = j.detail.map((d: { msg?: string; loc?: unknown[] }) => `${(d.loc ?? []).slice(1).join(" › ")}: ${d.msg}`);
    else if (typeof j?.detail === "string") errors = [j.detail];
  } catch {
    /* keep status text */
  }
  return new HttpError(res.status, errors.join(" "), errors);
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: "application/json" }, credentials: "same-origin" });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<T>;
}

/** State-changing request. The custom header is the server's CSRF check. */
async function send<T>(method: "POST" | "DELETE", path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", Accept: "application/json", "X-SepsisSense": "1" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw await parseError(res);
  return res.json() as Promise<T>;
}

// ------------------------------------------------------------------ auth

export function useSession() {
  return useQuery({
    queryKey: ["me"],
    queryFn: async () => {
      try {
        return (await get<{ user: SessionUser }>("/api/auth/me")).user;
      } catch (e) {
        if (e instanceof HttpError && e.status === 401) return null;
        throw e;
      }
    },
    staleTime: 5 * 60_000,
    retry: false,
  });
}

export function useLogin() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { username: string; password: string }) => send<{ user: SessionUser }>("POST", "/api/auth/login", body),
    onSuccess: (data) => {
      // Update the session in place (clear() would detach the observers watching it),
      // then drop every other cached query so nothing from a previous user leaks through.
      qc.setQueryData(["me"], data.user);
      qc.removeQueries({ predicate: (q) => q.queryKey[0] !== "me" });
    },
  });
}

export function useLogout() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => send<{ ok: boolean }>("POST", "/api/auth/logout"),
    onSettled: () => {
      qc.setQueryData(["me"], null);
      qc.removeQueries({ predicate: (q) => q.queryKey[0] !== "me" });
    },
  });
}

export function useDemoAccounts() {
  return useQuery({
    queryKey: ["demo-accounts"],
    queryFn: () => get<{ accounts: DemoAccount[] }>("/api/auth/demo-accounts"),
    staleTime: Infinity,
  });
}

// ---------------------------------------------------------------- replay

export function useReplay(poll = false) {
  return useQuery({
    queryKey: ["replay"],
    queryFn: () => get<{ offset: number; max_offset: number }>("/api/replay"),
    refetchInterval: poll ? 10_000 : false,
    staleTime: poll ? 0 : 30_000,
  });
}

export function useSetReplay() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (offset: number) => send<{ offset: number; max_offset: number }>("POST", "/api/replay", { offset }),
    onMutate: async (offset) => {
      await qc.cancelQueries({ queryKey: ["replay"] });
      const prev = qc.getQueryData<{ offset: number; max_offset: number }>(["replay"]);
      if (prev) qc.setQueryData(["replay"], { ...prev, offset });
      return { prev };
    },
    onError: (_e, _v, ctx) => ctx?.prev && qc.setQueryData(["replay"], ctx.prev),
    onSettled: () => qc.invalidateQueries({ queryKey: ["replay"] }),
  });
}

// ------------------------------------------------------------- clinical

export function useMeta() {
  return useQuery({
    queryKey: ["meta"],
    queryFn: () => get<Meta>("/api/meta"),
    staleTime: Infinity,
  });
}

export function useWard(offset: number | undefined) {
  return useQuery({
    queryKey: ["ward", offset],
    queryFn: () => get<WardResponse>("/api/ward"),
    enabled: offset !== undefined,
    placeholderData: keepPreviousData,
  });
}

export function usePatient(pid: string | undefined, offset: number | undefined, hour: number | null, poll = false) {
  const q = new URLSearchParams();
  if (hour !== null) q.set("hour", String(hour));
  return useQuery({
    queryKey: ["patient", pid, offset, hour],
    queryFn: () => get<PatientDetail>(`/api/patients/${pid}${q.size ? `?${q}` : ""}`),
    enabled: Boolean(pid) && offset !== undefined,
    placeholderData: keepPreviousData,
    refetchInterval: poll ? 30_000 : false,
  });
}

export function useAck(pid: string) {
  const qc = useQueryClient();
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["patient", pid] });
    qc.invalidateQueries({ queryKey: ["ward"] });
  };
  return {
    review: useMutation({
      mutationFn: (body: { reason: string; note: string }) => send<{ ack: ServerAck }>("POST", `/api/patients/${pid}/ack`, body),
      onSuccess: refresh,
    }),
    undo: useMutation({
      mutationFn: () => send<{ ok: boolean }>("DELETE", `/api/patients/${pid}/ack`),
      onSuccess: refresh,
    }),
  };
}

export function useAssessFields() {
  return useQuery({
    queryKey: ["assess-fields"],
    queryFn: () => get<AssessFieldsResponse>("/api/assess/fields"),
    staleTime: Infinity,
  });
}

export function useAssess() {
  return useMutation({
    mutationFn: (body: AssessRequest) => send<AssessResponse>("POST", "/api/assess", body),
  });
}

// ----------------------------------------------------------------- admin

export function useAdminUsers() {
  return useQuery({ queryKey: ["admin-users"], queryFn: () => get<AdminUsersResponse>("/api/admin/users") });
}

export function useAdminActions() {
  const qc = useQueryClient();
  const refresh = () => qc.invalidateQueries({ queryKey: ["admin-users"] });
  return {
    create: useMutation({
      mutationFn: (body: { username: string; display_name: string; role: "clinician" | "patient"; pid?: string | null }) =>
        send<{ user: SessionUser; temporary_password: string }>("POST", "/api/admin/users", body),
      onSuccess: refresh,
    }),
    reset: useMutation({
      mutationFn: (username: string) =>
        send<{ username: string; temporary_password: string }>("POST", `/api/admin/users/${username}/reset-password`),
      onSuccess: refresh,
    }),
    remove: useMutation({
      mutationFn: (username: string) => send<{ ok: boolean }>("DELETE", `/api/admin/users/${username}`),
      onSuccess: refresh,
    }),
  };
}
