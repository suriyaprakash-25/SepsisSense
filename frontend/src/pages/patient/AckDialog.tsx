import clsx from "clsx";
import { useEffect, useState } from "react";
import { Button, Dialog } from "../../components/ui";
import { ACK_REASONS, type AckReason } from "../../lib/store";

export function AckDialog({
  open,
  onClose,
  onConfirm,
  bed,
  since,
  pending = false,
  error = null,
}: {
  open: boolean;
  onClose: () => void;
  onConfirm: (reason: AckReason, note: string) => void;
  bed: string;
  since: number | null;
  pending?: boolean;
  error?: string | null;
}) {
  const [reason, setReason] = useState<AckReason | null>(null);
  const [note, setNote] = useState("");

  useEffect(() => {
    if (open) {
      setReason(null);
      setNote("");
    }
  }, [open]);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={`Review alert — ${bed}`}
      description={`Alert active since ICU hour ${since ?? "—"}. Your review is shared with the care team and shown to the patient as "reviewed by your care team" (reason and note stay with clinicians).`}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" disabled={!reason || pending} onClick={() => reason && onConfirm(reason, note.trim())}>
            {pending ? "Saving…" : "Mark as reviewed"}
          </Button>
        </>
      }
    >
      <fieldset>
        <legend className="mb-2 text-sm font-medium text-ink">Outcome of review</legend>
        <div className="space-y-2">
          {ACK_REASONS.map((r) => (
            <label
              key={r.id}
              className={clsx(
                "flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-colors",
                reason === r.id ? "border-brand bg-brand-soft" : "border-line hover:bg-hover",
              )}
            >
              <input
                type="radio"
                name="ack-reason"
                value={r.id}
                checked={reason === r.id}
                onChange={() => setReason(r.id)}
                className="mt-0.5 accent-[var(--brand)]"
              />
              <span>
                <span className="block text-sm font-medium text-ink">{r.label}</span>
                <span className="block text-xs text-muted">{r.hint}</span>
              </span>
            </label>
          ))}
        </div>
      </fieldset>
      <label className="mt-4 block">
        <span className="text-sm font-medium text-ink">
          Note <span className="font-normal text-faint">(optional)</span>
        </span>
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={2}
          maxLength={280}
          placeholder="e.g. Febrile post-op day 1, wound clean, will repeat lactate at 14:00"
          className="mt-1.5 w-full resize-none rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-faint focus:border-brand focus:outline-none"
        />
      </label>
      {error && <p className="mt-3 text-sm text-alert-ink">{error}</p>}
    </Dialog>
  );
}
