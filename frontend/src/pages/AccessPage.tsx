import clsx from "clsx";
import { Check, Copy, HeartPulse, KeyRound, ShieldCheck, Stethoscope, Trash2, UserPlus, Users } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Badge, Button, Card, CardHeader, Dialog, EmptyState, Segmented, Skeleton } from "../components/ui";
import { HttpError, useAdminActions, useAdminUsers } from "../lib/api";
import type { AdminUser } from "../lib/types";

function errText(e: unknown): string | null {
  if (!e) return null;
  return e instanceof HttpError ? e.message : String(e);
}

function CopyValue({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          /* clipboard unavailable */
        }
      }}
      className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-1.5 font-mono text-sm text-ink hover:bg-hover"
      title="Copy to clipboard"
    >
      {value}
      {copied ? <Check className="size-4 text-ok" /> : <Copy className="size-4 text-faint" />}
    </button>
  );
}

export function AccessPage() {
  const { data, isLoading, isError, error } = useAdminUsers();
  const { create, reset, remove } = useAdminActions();
  const [role, setRole] = useState<"patient" | "clinician">("patient");
  const [pid, setPid] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [username, setUsername] = useState("");
  const [issued, setIssued] = useState<{ username: string; password: string; kind: "created" | "reset" } | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<AdminUser | null>(null);

  const patientsWithAccount = useMemo(
    () => new Set((data?.users ?? []).filter((u) => u.role === "patient").map((u) => u.pid)),
    [data],
  );
  const freeBeds = (data?.patients ?? []).filter((p) => !patientsWithAccount.has(p.pid));

  // Suggest a username and display name from the chosen bed.
  useEffect(() => {
    if (role !== "patient") return;
    const bed = data?.patients.find((p) => p.pid === pid)?.bed;
    if (bed) {
      setDisplayName(`Patient in ${bed}`);
      setUsername(bed.toLowerCase().replace(/[^a-z0-9]/g, ""));
    }
  }, [pid, role, data]);

  const canCreate =
    /^[a-z0-9][a-z0-9._-]{2,31}$/.test(username) && displayName.trim().length > 0 && (role === "clinician" || Boolean(pid));

  const submit = () => {
    if (!canCreate) return;
    create.mutate(
      { username, display_name: displayName.trim(), role, pid: role === "patient" ? pid : null },
      {
        onSuccess: (r) => {
          setIssued({ username: r.user.username, password: r.temporary_password, kind: "created" });
          setPid("");
          setDisplayName("");
          setUsername("");
        },
      },
    );
  };

  return (
    <div className="mx-auto max-w-6xl space-y-5 px-4 py-6 sm:px-6">
      <div>
        <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-ink">
          <Users className="size-5 text-brand" /> Access management
        </h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Create sign-ins for patients and colleagues. A patient account is linked to one bed and can only ever see that
          record; clinician accounts see the whole ward and can manage access.
        </p>
      </div>

      {issued && (
        <Card className="border-ok/40 bg-ok-soft">
          <div className="flex flex-wrap items-center gap-4 p-4">
            <ShieldCheck className="size-6 shrink-0 text-ok" />
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold text-ink">
                {issued.kind === "created" ? "Account created" : "Password reset"} for{" "}
                <span className="font-mono">{issued.username}</span>
              </div>
              <div className="text-xs text-muted">
                Share this temporary password securely. It is shown only once and any existing sessions were signed out.
              </div>
            </div>
            <CopyValue value={issued.password} />
            <Button size="sm" variant="ghost" onClick={() => setIssued(null)}>
              Done
            </Button>
          </div>
        </Card>
      )}

      <div className="grid gap-5 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardHeader icon={<UserPlus className="size-4" />} title="New account" />
          <div className="space-y-4 px-4 pb-4">
            <Segmented
              label="Account type"
              value={role}
              onChange={(v) => {
                setRole(v);
                setPid("");
                setDisplayName("");
                setUsername("");
              }}
              options={[
                { value: "patient", label: "Patient" },
                { value: "clinician", label: "Clinician" },
              ]}
            />
            {role === "patient" && (
              <label className="block">
                <span className="mb-1.5 block text-[13px] font-medium text-ink">Patient (bed)</span>
                <select
                  value={pid}
                  onChange={(e) => setPid(e.target.value)}
                  className="h-9 w-full rounded-lg border border-line bg-surface px-2.5 text-sm text-ink focus:border-brand focus:outline-none"
                >
                  <option value="">Choose a bed…</option>
                  {freeBeds.map((p) => (
                    <option key={p.pid} value={p.pid}>
                      {p.bed} · {p.pid}
                    </option>
                  ))}
                </select>
                <span className="mt-1 block text-[11px] text-faint">
                  {freeBeds.length} of {data?.patients.length ?? 0} beds have no account yet
                </span>
              </label>
            )}
            <label className="block">
              <span className="mb-1.5 block text-[13px] font-medium text-ink">Display name</span>
              <input
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder={role === "patient" ? "Patient in MICU-03" : "Dr. Rivera"}
                className="h-9 w-full rounded-lg border border-line bg-surface px-3 text-sm text-ink focus:border-brand focus:outline-none"
              />
            </label>
            <label className="block">
              <span className="mb-1.5 block text-[13px] font-medium text-ink">Username</span>
              <input
                value={username}
                onChange={(e) => setUsername(e.target.value.toLowerCase())}
                placeholder={role === "patient" ? "micu03" : "rivera"}
                autoCapitalize="none"
                spellCheck={false}
                className="h-9 w-full rounded-lg border border-line bg-surface px-3 font-mono text-sm text-ink focus:border-brand focus:outline-none"
              />
              <span className="mt-1 block text-[11px] text-faint">3–32 characters: lowercase letters, numbers, . _ -</span>
            </label>
            {create.error && <p className="text-sm text-alert-ink">{errText(create.error)}</p>}
            <Button variant="primary" className="w-full" disabled={!canCreate || create.isPending} onClick={submit}>
              {create.isPending ? "Creating…" : "Create account"}
            </Button>
            <p className="text-[11px] leading-4 text-faint">A temporary password is generated and shown once.</p>
          </div>
        </Card>

        <Card className="overflow-hidden lg:col-span-2">
          <CardHeader icon={<Users className="size-4" />} title="Accounts" subtitle={data ? `${data.users.length} accounts` : undefined} />
          {isError ? (
            <EmptyState icon={<Users className="size-5" />} title="Could not load accounts">
              {errText(error)}
            </EmptyState>
          ) : isLoading || !data ? (
            <div className="space-y-2 p-4">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-12" />
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-y border-line bg-subtle/60 text-[11px] font-semibold tracking-wide text-faint uppercase">
                    <th className="px-4 py-2 text-left">Name</th>
                    <th className="px-3 py-2 text-left">Role</th>
                    <th className="px-3 py-2 text-left">Can see</th>
                    <th className="px-3 py-2 text-left">Last sign-in</th>
                    <th className="px-4 py-2 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {data.users.map((u) => (
                    <tr key={u.username} className="border-b border-line last:border-0">
                      <td className="px-4 py-2.5">
                        <div className="font-medium text-ink">
                          {u.display_name} {u.username === data.me && <span className="text-xs font-normal text-faint">(you)</span>}
                        </div>
                        <div className="font-mono text-xs text-muted">{u.username}</div>
                      </td>
                      <td className="px-3 py-2.5">
                        {u.role === "clinician" ? (
                          <Badge tone="brand" icon={<Stethoscope className="size-3" />}>
                            Clinician · admin
                          </Badge>
                        ) : (
                          <Badge tone="ok" icon={<HeartPulse className="size-3" />}>
                            Patient
                          </Badge>
                        )}
                      </td>
                      <td className="px-3 py-2.5 text-[13px] text-muted">
                        {u.role === "clinician" ? "All patients" : u.bed ? `Own record · ${u.bed}` : "—"}
                      </td>
                      <td className="px-3 py-2.5 text-xs text-muted tnum">
                        {u.last_login ? new Date(u.last_login).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "Never"}
                      </td>
                      <td className="px-4 py-2.5">
                        <div className="flex justify-end gap-1">
                          <Button
                            size="sm"
                            variant="ghost"
                            icon={<KeyRound className="size-3.5" />}
                            disabled={reset.isPending}
                            onClick={() =>
                              reset.mutate(u.username, {
                                onSuccess: (r) => setIssued({ username: r.username, password: r.temporary_password, kind: "reset" }),
                              })
                            }
                          >
                            Reset password
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            icon={<Trash2 className="size-3.5" />}
                            disabled={u.username === data.me}
                            title={u.username === data.me ? "You cannot delete your own account" : "Delete account"}
                            onClick={() => setConfirmDelete(u)}
                            className={clsx(u.username !== data.me && "hover:text-alert-ink")}
                          >
                            <span className="sr-only">Delete</span>
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {reset.error && <p className="px-4 pb-3 text-sm text-alert-ink">{errText(reset.error)}</p>}
            </div>
          )}
        </Card>
      </div>

      <Dialog
        open={confirmDelete !== null}
        onClose={() => {
          setConfirmDelete(null);
          remove.reset();
        }}
        title={`Delete ${confirmDelete?.display_name ?? ""}?`}
        description="They will be signed out and can no longer sign in. Their clinical record is not affected."
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirmDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              disabled={remove.isPending}
              onClick={() =>
                confirmDelete &&
                remove.mutate(confirmDelete.username, {
                  onSuccess: () => setConfirmDelete(null),
                })
              }
            >
              {remove.isPending ? "Deleting…" : "Delete account"}
            </Button>
          </>
        }
      >
        <p className="font-mono text-sm text-muted">{confirmDelete?.username}</p>
        {remove.error && <p className="mt-2 text-sm text-alert-ink">{errText(remove.error)}</p>}
      </Dialog>
    </div>
  );
}
