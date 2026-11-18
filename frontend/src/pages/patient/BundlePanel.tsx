import clsx from "clsx";
import { Check, RotateCcw } from "lucide-react";
import { useBundle } from "../../lib/store";
import type { BundleItem } from "../../lib/types";

export function BundlePanel({ pid, items }: { pid: string; items: BundleItem[] }) {
  const [done, setDone] = useBundle(pid);
  const count = items.filter((i) => done[i.id]).length;

  const toggle = (id: string) =>
    setDone((prev) => {
      const next = { ...prev };
      if (next[id]) delete next[id];
      else next[id] = new Date().toISOString();
      return next;
    });

  return (
    <div className="px-4 pb-4">
      <div className="mb-3 flex items-center gap-3">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-subtle">
          <div className="h-full rounded-full bg-ok transition-[width]" style={{ width: `${(count / items.length) * 100}%` }} />
        </div>
        <span className="text-xs font-medium text-muted tnum">
          {count}/{items.length} done
        </span>
        {count > 0 && (
          <button
            onClick={() => setDone({})}
            className="text-faint hover:text-ink"
            aria-label="Clear checklist"
            title="Clear checklist"
          >
            <RotateCcw className="size-3.5" />
          </button>
        )}
      </div>
      <ul className="space-y-1.5">
        {items.map((it) => {
          const checked = Boolean(done[it.id]);
          return (
            <li key={it.id}>
              <label
                className={clsx(
                  "flex cursor-pointer gap-3 rounded-lg border p-2.5 transition-colors",
                  checked
                    ? "border-line bg-subtle"
                    : it.priority === "high"
                      ? "border-warn-line bg-warn-soft/40 hover:bg-warn-soft/70"
                      : "border-line hover:bg-hover",
                )}
              >
                <input type="checkbox" className="peer sr-only" checked={checked} onChange={() => toggle(it.id)} />
                <span
                  className={clsx(
                    "mt-0.5 flex size-[18px] shrink-0 items-center justify-center rounded-md border transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-brand",
                    checked ? "border-ok bg-ok text-white" : "border-line-strong bg-surface",
                  )}
                >
                  {checked && <Check className="size-3" strokeWidth={3} />}
                </span>
                <span className="min-w-0">
                  <span className={clsx("block text-[13px] font-medium", checked ? "text-muted line-through" : "text-ink")}>
                    {it.title}
                  </span>
                  <span className="mt-0.5 block text-xs text-muted">{it.context}</span>
                  {!checked && <span className="mt-0.5 block text-[11px] text-faint">{it.detail}</span>}
                  {checked && (
                    <span className="mt-0.5 block text-[11px] text-faint">
                      Marked at {new Date(done[it.id]).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </span>
                  )}
                </span>
              </label>
            </li>
          );
        })}
      </ul>
      <p className="mt-3 text-[11px] leading-4 text-faint">
        Surviving Sepsis Campaign 2021 reminders, annotated with what is on the chart. Rule-based, not model output.
        Checklist state is saved in this browser only.
      </p>
    </div>
  );
}
