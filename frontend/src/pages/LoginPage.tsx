import clsx from "clsx";
import { Activity, Eye, EyeOff, HeartPulse, LockKeyhole, ShieldAlert, Stethoscope } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Button } from "../components/ui";
import { HttpError, useDemoAccounts, useLogin } from "../lib/api";

export function LoginPage() {
  const login = useLogin();
  const demo = useDemoAccounts();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password) return;
    login.mutate({ username: username.trim().toLowerCase(), password });
  };

  const error = login.error instanceof HttpError ? login.error.message : login.error ? "Could not reach the server." : null;

  return (
    <div className="flex min-h-full">
      {/* Brand panel */}
      <aside className="relative hidden w-[44%] flex-col justify-between overflow-hidden bg-[#0f2a6b] p-10 text-white lg:flex">
        <div className="pointer-events-none absolute -top-24 -right-24 size-96 rounded-full bg-white/5" />
        <div className="pointer-events-none absolute -bottom-32 -left-16 size-[28rem] rounded-full bg-white/5" />
        <div className="relative flex items-center gap-3">
          <span className="flex size-10 items-center justify-center rounded-xl bg-white text-[#1e4fd8]">
            <Activity className="size-5" strokeWidth={2.4} />
          </span>
          <div>
            <div className="text-lg font-semibold tracking-tight">SepsisSense</div>
            <div className="text-sm text-white/70">Sepsis early warning</div>
          </div>
        </div>
        <div className="relative max-w-md">
          <h1 className="text-3xl leading-tight font-semibold tracking-tight">
            One record, two views: the care team and the patient.
          </h1>
          <ul className="mt-8 space-y-5">
            <li className="flex gap-3">
              <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-white/10">
                <Stethoscope className="size-[18px]" />
              </span>
              <span>
                <span className="block font-medium">Clinicians</span>
                <span className="block text-sm text-white/70">
                  Whole-ward risk overview, alerts to review, bedside assessment, and patient access management.
                </span>
              </span>
            </li>
            <li className="flex gap-3">
              <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-white/10">
                <HeartPulse className="size-[18px]" />
              </span>
              <span>
                <span className="block font-medium">Patients</span>
                <span className="block text-sm text-white/70">
                  Their own vital signs, lab results and sepsis outlook, explained in plain language.
                </span>
              </span>
            </li>
          </ul>
        </div>
        <p className="relative flex items-start gap-2 text-xs text-white/60">
          <ShieldAlert className="mt-0.5 size-3.5 shrink-0" />
          Research prototype. Replays de-identified PhysioNet 2019 ICU stays; not for clinical use.
        </p>
      </aside>

      {/* Form */}
      <main className="flex flex-1 items-center justify-center px-4 py-10 sm:px-8">
        <div className="w-full max-w-sm">
          <div className="mb-8 flex items-center gap-2.5 lg:hidden">
            <span className="flex size-9 items-center justify-center rounded-lg bg-brand text-white">
              <Activity className="size-[18px]" strokeWidth={2.4} />
            </span>
            <span className="text-lg font-semibold tracking-tight text-ink">SepsisSense</span>
          </div>
          <h2 className="text-2xl font-semibold tracking-tight text-ink">Sign in</h2>
          <p className="mt-1 text-sm text-muted">Use the account your care team gave you.</p>

          <form onSubmit={submit} className="mt-6 space-y-4" noValidate>
            <label className="block">
              <span className="mb-1.5 block text-sm font-medium text-ink">Username</span>
              <input
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                autoCapitalize="none"
                spellCheck={false}
                autoFocus
                className="h-10 w-full rounded-lg border border-line bg-surface px-3 text-sm text-ink shadow-card focus:border-brand focus:outline-none"
              />
            </label>
            <label className="block">
              <span className="mb-1.5 block text-sm font-medium text-ink">Password</span>
              <span className="relative block">
                <input
                  type={show ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  className="h-10 w-full rounded-lg border border-line bg-surface pr-10 pl-3 text-sm text-ink shadow-card focus:border-brand focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => setShow(!show)}
                  className="absolute top-1/2 right-2 -translate-y-1/2 rounded p-1 text-faint hover:text-ink"
                  aria-label={show ? "Hide password" : "Show password"}
                >
                  {show ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                </button>
              </span>
            </label>
            {error && (
              <p role="alert" className="rounded-lg bg-alert-soft px-3 py-2 text-sm text-alert-ink">
                {error}
              </p>
            )}
            <Button
              type="submit"
              variant="primary"
              className="h-10 w-full"
              disabled={login.isPending || !username.trim() || !password}
              icon={<LockKeyhole className="size-4" />}
            >
              {login.isPending ? "Signing in…" : "Sign in"}
            </Button>
          </form>

          {demo.data && demo.data.accounts.length > 0 && (
            <div className="mt-8 rounded-xl border border-dashed border-line-strong p-4">
              <div className="text-xs font-semibold tracking-wide text-faint uppercase">Demo accounts</div>
              <p className="mt-1 text-xs text-muted">Local research demo only. Click to fill in.</p>
              <ul className="mt-3 space-y-1.5">
                {demo.data.accounts.map((a) => (
                  <li key={a.username}>
                    <button
                      type="button"
                      onClick={() => {
                        setUsername(a.username);
                        setPassword(a.password);
                        login.reset();
                      }}
                      className={clsx(
                        "flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors hover:bg-hover",
                        username === a.username ? "border-brand bg-brand-soft" : "border-line",
                      )}
                    >
                      <span
                        className={clsx(
                          "flex size-8 shrink-0 items-center justify-center rounded-lg",
                          a.role === "clinician" ? "bg-brand-soft text-brand-ink" : "bg-ok-soft text-ok",
                        )}
                      >
                        {a.role === "clinician" ? <Stethoscope className="size-4" /> : <HeartPulse className="size-4" />}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-medium text-ink">{a.display_name}</span>
                        <span className="block text-xs text-muted">
                          {a.role === "clinician" ? "Clinician (admin)" : "Patient"} ·{" "}
                          <span className="font-mono">{a.username}</span> / <span className="font-mono">{a.password}</span>
                        </span>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
