import clsx from "clsx";
import {
  Activity,
  BookOpenCheck,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
  Keyboard,
  LayoutGrid,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  RotateCcw,
  Settings2,
  ShieldAlert,
  Sun,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useMeta } from "../lib/api";
import { usePrefs } from "../lib/prefs";
import { Dialog, IconButton, Kbd, Popover, Switch, Tooltip } from "./ui";

const NAV = [
  { to: "/", label: "Ward overview", icon: LayoutGrid, end: true },
  { to: "/model", label: "Model facts", icon: BookOpenCheck, end: false },
];

export function AppShell({ children }: { children: ReactNode }) {
  const { sidebarCollapsed, setSidebarCollapsed, theme, setTheme, offset, setOffset } = usePrefs();
  const meta = useMeta();
  const maxOffset = meta.data?.max_offset ?? 24;
  const [helpOpen, setHelpOpen] = useState(false);
  const navigate = useNavigate();

  // Global shortcuts (ignored while typing)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, [contenteditable=true]") || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "?") setHelpOpen(true);
      else if (e.key === "]") setOffset((o) => Math.min(maxOffset, o + 1));
      else if (e.key === "[") setOffset((o) => Math.max(0, o - 1));
      else if (e.key === "t") setTheme(theme === "dark" ? "light" : "dark");
      else if (e.key === "g") navigate("/");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [maxOffset, setOffset, setTheme, theme, navigate]);

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <aside
        className={clsx(
          "hidden shrink-0 flex-col border-r border-line bg-surface transition-[width] duration-200 md:flex",
          sidebarCollapsed ? "w-16" : "w-60",
        )}
      >
        <div className={clsx("flex h-14 items-center gap-2.5 border-b border-line", sidebarCollapsed ? "justify-center" : "px-4")}>
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-brand text-white">
            <Activity className="size-[18px]" strokeWidth={2.4} />
          </span>
          {!sidebarCollapsed && (
            <div className="leading-tight">
              <div className="text-[15px] font-semibold tracking-tight text-ink">SepsisSense</div>
              <div className="text-[11px] text-faint">Sepsis early warning</div>
            </div>
          )}
        </div>
        <nav className="flex-1 space-y-0.5 p-2" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              title={sidebarCollapsed ? label : undefined}
              className={({ isActive }) =>
                clsx(
                  "flex h-9 items-center gap-2.5 rounded-lg text-sm font-medium transition-colors",
                  sidebarCollapsed ? "justify-center" : "px-3",
                  isActive ? "bg-brand-soft text-brand-ink" : "text-muted hover:bg-hover hover:text-ink",
                )
              }
            >
              <Icon className="size-[18px] shrink-0" />
              {!sidebarCollapsed && label}
            </NavLink>
          ))}
        </nav>
        <div className="space-y-0.5 border-t border-line p-2">
          <button
            onClick={() => setHelpOpen(true)}
            className={clsx(
              "flex h-9 w-full items-center gap-2.5 rounded-lg text-sm text-muted hover:bg-hover hover:text-ink",
              sidebarCollapsed ? "justify-center" : "px-3",
            )}
            title="Keyboard shortcuts"
          >
            <Keyboard className="size-[18px] shrink-0" />
            {!sidebarCollapsed && (
              <>
                Shortcuts <span className="ml-auto"><Kbd>?</Kbd></span>
              </>
            )}
          </button>
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className={clsx(
              "flex h-9 w-full items-center gap-2.5 rounded-lg text-sm text-muted hover:bg-hover hover:text-ink",
              sidebarCollapsed ? "justify-center" : "px-3",
            )}
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? <PanelLeftOpen className="size-[18px]" /> : <PanelLeftClose className="size-[18px]" />}
            {!sidebarCollapsed && "Collapse"}
          </button>
        </div>
      </aside>

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 border-b border-warn-line bg-warn-soft px-4 py-1.5 text-xs text-warn-ink">
          <ShieldAlert className="size-3.5 shrink-0" />
          <span className="min-w-0 truncate">
            <strong className="font-semibold">Research prototype — not for clinical use.</strong>{" "}
            <span className="hidden sm:inline">Replaying de-identified ICU stays from the public PhysioNet 2019 dataset.</span>
          </span>
        </div>
        <header className="flex h-14 shrink-0 items-center gap-2 border-b border-line bg-surface px-3 sm:px-4">
          <div className="flex items-center gap-2 md:hidden">
            <span className="flex size-8 items-center justify-center rounded-lg bg-brand text-white">
              <Activity className="size-[18px]" strokeWidth={2.4} />
            </span>
            <nav className="flex gap-1" aria-label="Main mobile">
              {NAV.map(({ to, label, icon: Icon, end }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={end}
                  aria-label={label}
                  className={({ isActive }) =>
                    clsx("flex size-9 items-center justify-center rounded-lg", isActive ? "bg-brand-soft text-brand-ink" : "text-muted")
                  }
                >
                  <Icon className="size-[18px]" />
                </NavLink>
              ))}
            </nav>
          </div>

          <ReplayClock offset={offset} setOffset={setOffset} max={maxOffset} />

          <div className="ml-auto flex items-center gap-1">
            <IconButton
              label={theme === "dark" ? "Switch to light theme (T)" : "Switch to dark theme for night shift (T)"}
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? <Sun className="size-[18px]" /> : <Moon className="size-[18px]" />}
            </IconButton>
            <SettingsMenu />
          </div>
        </header>
        <main className="min-h-0 flex-1 overflow-y-auto scrollbar-thin">{children}</main>
      </div>

      <ShortcutsDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  );
}

function ReplayClock({
  offset,
  setOffset,
  max,
}: {
  offset: number;
  setOffset: (v: number | ((p: number) => number)) => void;
  max: number;
}) {
  return (
    <div className="flex items-center gap-1 rounded-lg border border-line bg-subtle p-0.5">
      <Tooltip
        side="bottom"
        content="Recorded stays are replayed hour by hour. Step forward to watch risk evolve as new data arrives. Shortcuts: [ and ]"
      >
        <span className="hidden items-center gap-1.5 pr-1 pl-2 text-xs font-medium text-muted sm:inline-flex">
          <FlaskConical className="size-3.5" />
          Replay
        </span>
      </Tooltip>
      <IconButton
        label="Back 1 hour ([)"
        className="size-7"
        disabled={offset <= 0}
        onClick={() => setOffset((o) => Math.max(0, o - 1))}
      >
        <ChevronLeft className="size-4" />
      </IconButton>
      <span className="min-w-[4.5rem] text-center text-[13px] font-semibold text-ink tnum" aria-live="polite">
        {offset === 0 ? "Start" : `+${offset} h`}
      </span>
      <IconButton
        label="Forward 1 hour (])"
        className="size-7"
        disabled={offset >= max}
        onClick={() => setOffset((o) => Math.min(max, o + 1))}
      >
        <ChevronRight className="size-4" />
      </IconButton>
      {offset > 0 && (
        <IconButton label="Reset replay to start" className="size-7" onClick={() => setOffset(0)}>
          <RotateCcw className="size-3.5" />
        </IconButton>
      )}
    </div>
  );
}

function SettingsMenu() {
  const { research, setResearch, theme, setTheme } = usePrefs();
  return (
    <Popover
      trigger={({ open, toggle }) => (
        <IconButton label="Display settings" active={open} onClick={toggle}>
          <Settings2 className="size-[18px]" />
        </IconButton>
      )}
    >
      <div className="space-y-4">
        <div className="text-xs font-semibold tracking-wide text-faint uppercase">Display</div>
        <Switch
          checked={theme === "dark"}
          onChange={(v) => setTheme(v ? "dark" : "light")}
          label="Night mode"
          description="Low-glare palette for dim rooms."
        />
        <Switch
          checked={research}
          onChange={setResearch}
          label="Show retrospective labels"
          description="Reveal when each stay met Sepsis-3 criteria. For model evaluation only — a clinician would not have this."
        />
      </div>
    </Popover>
  );
}

const SHORTCUTS: [string[], string][] = [
  [["/"], "Search patients"],
  [["J", "K"], "Move down / up the ward list"],
  [["Enter"], "Open selected patient"],
  [["Esc"], "Back to ward from a patient"],
  [["[", "]"], "Replay back / forward one hour"],
  [["A"], "Acknowledge the open patient's alert"],
  [["T"], "Toggle night mode"],
  [["G"], "Go to ward overview"],
  [["?"], "Show this list"],
];

function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Dialog open={open} onClose={onClose} title="Keyboard shortcuts" description="Designed for one-handed use at the workstation.">
      <ul className="divide-y divide-line">
        {SHORTCUTS.map(([keys, label]) => (
          <li key={label} className="flex items-center justify-between py-2 text-sm">
            <span className="text-ink">{label}</span>
            <span className="flex gap-1">
              {keys.map((k) => (
                <Kbd key={k}>{k}</Kbd>
              ))}
            </span>
          </li>
        ))}
      </ul>
    </Dialog>
  );
}
