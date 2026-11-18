import { useCallback, useSyncExternalStore } from "react";

/**
 * Tiny localStorage-backed store. Every read/write is guarded: storage can be
 * unavailable (private windows, locked-down hospital browsers) and the UI must
 * still work, just without persistence.
 */
const listeners = new Set<() => void>();
const memory = new Map<string, string>();

function read(key: string): string | null {
  try {
    const v = window.localStorage.getItem(key);
    if (v !== null) return v;
  } catch {
    /* storage unavailable */
  }
  return memory.get(key) ?? null;
}

function write(key: string, value: string | null) {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable */
  }
  if (value === null) memory.delete(key);
  else memory.set(key, value);
  listeners.forEach((l) => l());
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  const onStorage = () => cb();
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(cb);
    window.removeEventListener("storage", onStorage);
  };
}

export function usePersisted<T>(key: string, fallback: T): [T, (v: T | ((prev: T) => T)) => void] {
  const raw = useSyncExternalStore(subscribe, () => read(key), () => null);
  let value: T = fallback;
  if (raw !== null) {
    try {
      value = JSON.parse(raw) as T;
    } catch {
      value = fallback;
    }
  }
  const set = useCallback(
    (v: T | ((prev: T) => T)) => {
      const prevRaw = read(key);
      let prev = fallback;
      if (prevRaw !== null) {
        try {
          prev = JSON.parse(prevRaw) as T;
        } catch {
          /* ignore */
        }
      }
      const next = typeof v === "function" ? (v as (p: T) => T)(prev) : v;
      write(key, JSON.stringify(next));
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [key],
  );
  return [value, set];
}

// ---------------------------------------------------------------- domain stores

export type AckReason = "workup" | "alternative" | "known" | "comfort";

export const ACK_REASONS: { id: AckReason; label: string; hint: string }[] = [
  { id: "workup", label: "Sepsis suspected — work-up started", hint: "Cultures, lactate or antibiotics ordered" },
  { id: "alternative", label: "Reviewed — alternative explanation", hint: "e.g. post-op, drug fever, pain, agitation" },
  { id: "known", label: "Already being treated for infection", hint: "Alert adds no new information" },
  { id: "comfort", label: "Comfort-focused care", hint: "Escalation not consistent with goals of care" },
];

export function useBundle(pid: string) {
  return usePersisted<Record<string, string>>(`ss.bundle.${pid}`, {});
}
