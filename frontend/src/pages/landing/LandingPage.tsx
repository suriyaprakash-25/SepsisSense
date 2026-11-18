import clsx from "clsx";
import {
  Activity,
  ArrowRight,
  BellRing,
  BrainCircuit,
  ChevronDown,
  CircleCheck,
  ClipboardCheck,
  ClipboardPlus,
  Clock3,
  Database,
  EyeOff,
  FlaskConical,
  Github,
  HandHeart,
  HeartPulse,
  LayoutGrid,
  LockKeyhole,
  Menu,
  MessageCircleQuestion,
  Moon,
  Scale,
  ShieldCheck,
  Siren,
  Sparkles,
  Stethoscope,
  Sun,
  Timer,
  Users,
  X,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { usePublicSummary } from "../../lib/api";
import { usePrefs } from "../../lib/prefs";
import { HeroPreview } from "./HeroPreview";

// ------------------------------------------------------------- helpers

/** Fade sections up as they enter the viewport (skipped for reduced motion). */
function useReveal() {
  useEffect(() => {
    const els = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));
    if (!("IntersectionObserver" in window) || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      els.forEach((e) => e.classList.add("is-visible"));
      return;
    }
    const io = new IntersectionObserver(
      (entries) =>
        entries.forEach((en) => {
          if (en.isIntersecting) {
            en.target.classList.add("is-visible");
            io.unobserve(en.target);
          }
        }),
      { rootMargin: "0px 0px -10% 0px" },
    );
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
}

function Section({ id, className, children }: { id?: string; className?: string; children: ReactNode }) {
  return (
    <section id={id} className={clsx("scroll-mt-20 px-5 sm:px-8", className)}>
      <div className="mx-auto max-w-6xl">{children}</div>
    </section>
  );
}

function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="text-[13px] font-semibold tracking-wide text-brand uppercase">{children}</p>;
}

function H2({ children, className }: { children: ReactNode; className?: string }) {
  return <h2 className={clsx("mt-3 text-3xl font-semibold tracking-tight text-balance text-ink sm:text-4xl", className)}>{children}</h2>;
}

const pctFmt = (v: number | null | undefined, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);

const NAV = [
  { href: "#how", label: "How it works" },
  { href: "#roles", label: "Clinicians & patients" },
  { href: "#evidence", label: "Evidence" },
  { href: "#faq", label: "FAQ" },
];

// ---------------------------------------------------------------- page

export function LandingPage() {
  useReveal();
  return (
    <div className="min-h-full bg-canvas">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-lg focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </a>
      <TopNav />
      <main id="main">
        <Hero />
        <Problem />
        <HowItWorks />
        <Roles />
        <Principles />
        <Evidence />
        <Trust />
        <Faq />
        <FinalCta />
      </main>
      <Footer />
    </div>
  );
}

// ------------------------------------------------------------------ nav

function TopNav() {
  const { theme, setTheme } = usePrefs();
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  return (
    <header
      className={clsx(
        "sticky top-0 z-40 transition-colors",
        scrolled || open ? "border-b border-line bg-surface/90 backdrop-blur-md" : "border-b border-transparent",
      )}
    >
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-6 px-5 sm:px-8">
        <a href="#top" className="flex items-center gap-2.5" aria-label="SepsisSense home">
          <span className="flex size-8 items-center justify-center rounded-lg bg-brand text-white">
            <Activity className="size-[18px]" strokeWidth={2.4} />
          </span>
          <span className="text-[17px] font-semibold tracking-tight text-ink">SepsisSense</span>
        </a>
        <nav className="hidden items-center gap-1 md:flex" aria-label="Sections">
          {NAV.map((n) => (
            <a key={n.href} href={n.href} className="rounded-lg px-3 py-2 text-sm font-medium text-muted hover:text-ink">
              {n.label}
            </a>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-1.5">
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="flex size-9 items-center justify-center rounded-lg text-muted hover:bg-hover hover:text-ink"
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          >
            {theme === "dark" ? <Sun className="size-[18px]" /> : <Moon className="size-[18px]" />}
          </button>
          <Link
            to="/login"
            className="hidden h-9 items-center gap-1.5 rounded-lg bg-brand px-4 text-sm font-semibold text-white hover:bg-brand-hover sm:inline-flex"
          >
            Sign in <ArrowRight className="size-4" />
          </Link>
          <button
            onClick={() => setOpen(!open)}
            className="flex size-9 items-center justify-center rounded-lg text-muted hover:bg-hover md:hidden"
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </div>
      </div>
      {open && (
        <nav className="border-t border-line px-5 pb-4 md:hidden" aria-label="Sections mobile">
          {NAV.map((n) => (
            <a key={n.href} href={n.href} onClick={() => setOpen(false)} className="block py-2.5 text-[15px] font-medium text-ink">
              {n.label}
            </a>
          ))}
          <Link to="/login" className="mt-2 flex h-10 items-center justify-center rounded-lg bg-brand text-sm font-semibold text-white">
            Sign in
          </Link>
        </nav>
      )}
    </header>
  );
}

// ----------------------------------------------------------------- hero

function Hero() {
  return (
    <Section id="top" className="relative overflow-hidden pt-12 pb-24 sm:pt-20 lg:pb-32">
      <div className="landing-grid pointer-events-none absolute inset-0 -z-0" aria-hidden />
      <div className="relative grid items-center gap-16 lg:grid-cols-[1.05fr_1fr] lg:gap-10">
        <div>
          <span className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1 text-xs font-medium text-muted shadow-card">
            <FlaskConical className="size-3.5 text-brand" />
            Research prototype · PhysioNet 2019 ICU data
          </span>
          <h1 className="mt-6 text-4xl leading-[1.08] font-semibold tracking-tight text-balance text-ink sm:text-5xl lg:text-[3.4rem]">
            An early-warning system for sepsis <span className="text-brand">that explains itself.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-8 text-muted">
            SepsisSense reads ICU vital signs and lab results every hour, estimates each patient's chance of sepsis in the
            next six hours, and shows why — to the care team in clinical terms, and to patients in plain language.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link
              to="/login"
              className="inline-flex h-12 items-center gap-2 rounded-xl bg-brand px-6 text-[15px] font-semibold text-white shadow-card hover:bg-brand-hover"
            >
              Sign in <ArrowRight className="size-4" />
            </Link>
            <a
              href="#how"
              className="inline-flex h-12 items-center gap-2 rounded-xl border border-line bg-surface px-6 text-[15px] font-semibold text-ink shadow-card hover:bg-hover"
            >
              See how it works
            </a>
          </div>
          <dl className="mt-10 grid max-w-lg grid-cols-3 gap-6 border-t border-line pt-6">
            {[
              { k: "Updates", v: "Hourly" },
              { k: "Looks ahead", v: "~6 hours" },
              { k: "Views", v: "Care team + patient" },
            ].map((s) => (
              <div key={s.k}>
                <dt className="text-xs text-faint">{s.k}</dt>
                <dd className="mt-1 text-sm font-semibold text-ink">{s.v}</dd>
              </div>
            ))}
          </dl>
        </div>
        <div className="pb-4 sm:pb-32 lg:pb-24">
          <HeroPreview />
        </div>
      </div>
    </Section>
  );
}

// -------------------------------------------------------------- problem

function Problem() {
  const stats = [
    {
      v: "48.9 million",
      k: "sepsis cases worldwide in 2017, with 11 million deaths — about 1 in 5 of all deaths.",
      src: "Rudd et al., The Lancet, 2020",
    },
    {
      v: "Every hour",
      k: "of delay to antibiotics in septic shock was linked to a 7.6% fall in survival.",
      src: "Kumar et al., Critical Care Medicine, 2006",
    },
    {
      v: "72–99%",
      k: "of clinical alarms are false, so teams learn to tune them out. Noisy sepsis alerts make this worse.",
      src: "Sendelbach & Funk, AACN Advanced Critical Care, 2013",
    },
  ];
  return (
    <Section className="border-y border-line bg-surface py-20 sm:py-24">
      <div data-reveal className="reveal max-w-2xl">
        <Eyebrow>Why it matters</Eyebrow>
        <H2>Sepsis is common, fast, and easy to miss in a busy ICU.</H2>
        <p className="mt-4 text-lg leading-8 text-muted">
          Catching it early saves lives, but early signs hide in hundreds of hourly numbers. Warning tools help only if
          clinicians trust them, and most generate so many false alarms that they are ignored.
        </p>
      </div>
      <div className="mt-12 grid gap-5 md:grid-cols-3">
        {stats.map((s, i) => (
          <figure key={s.v} data-reveal className="reveal rounded-2xl border border-line bg-canvas p-6" style={{ transitionDelay: `${i * 80}ms` }}>
            <div className="text-3xl font-semibold tracking-tight text-ink">{s.v}</div>
            <p className="mt-3 text-[15px] leading-7 text-muted">{s.k}</p>
            <figcaption className="mt-4 text-xs text-faint">{s.src}</figcaption>
          </figure>
        ))}
      </div>
    </Section>
  );
}

// ---------------------------------------------------------- how it works

function HowItWorks() {
  const steps: { icon: LucideIcon; title: string; body: string; detail: string }[] = [
    {
      icon: Database,
      title: "Reads the chart every hour",
      body: "Vital signs, blood tests, age and time in the ICU — the data already recorded at the bedside.",
      detail: "67 inputs, including 6-hour averages and 3-hour trends",
    },
    {
      icon: BrainCircuit,
      title: "Estimates the risk",
      body: "Three machine-learning models vote on each patient's chance of sepsis in roughly the next six hours.",
      detail: "LightGBM + XGBoost + Random forest, calibrated",
    },
    {
      icon: Sparkles,
      title: "Explains the score",
      body: "Every number comes with the readings that pushed it up or down, grouped into terms clinicians use.",
      detail: "TreeSHAP explanations per patient, per hour",
    },
    {
      icon: BellRing,
      title: "Alerts only when it's confident",
      body: "Alerts need two consecutive high hours. A clinician reviews each one, and the patient sees that it was reviewed.",
      detail: "Precision-tuned threshold + Hour-1 bundle checklist",
    },
  ];
  return (
    <Section id="how" className="py-20 sm:py-28">
      <div data-reveal className="reveal mx-auto max-w-2xl text-center">
        <Eyebrow>How it works</Eyebrow>
        <H2>From bedside numbers to a decision, in four steps.</H2>
      </div>
      <ol className="relative mt-14 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <div className="absolute top-7 right-[12%] left-[12%] hidden h-px bg-gradient-to-r from-transparent via-line-strong to-transparent lg:block" aria-hidden />
        {steps.map((s, i) => (
          <li key={s.title} data-reveal className="reveal relative" style={{ transitionDelay: `${i * 90}ms` }}>
            <div className="relative flex size-14 items-center justify-center rounded-2xl border border-line bg-surface shadow-card">
              <s.icon className="size-6 text-brand" />
              <span className="absolute -top-2 -right-2 flex size-6 items-center justify-center rounded-full bg-brand text-[11px] font-bold text-white">
                {i + 1}
              </span>
            </div>
            <h3 className="mt-5 text-lg font-semibold text-ink">{s.title}</h3>
            <p className="mt-2 text-[15px] leading-7 text-muted">{s.body}</p>
            <p className="mt-3 inline-block rounded-md bg-subtle px-2 py-1 text-xs font-medium text-muted">{s.detail}</p>
          </li>
        ))}
      </ol>
    </Section>
  );
}

// --------------------------------------------------------------- roles

function Roles() {
  const [role, setRole] = useState<"clinician" | "patient">("clinician");
  const content = {
    clinician: {
      title: "For the care team",
      lead: "Built for rounds and the central station: the whole ward on one screen, sorted by who needs attention first.",
      items: [
        { icon: LayoutGrid, t: "Ward overview", d: "Risk, 12-hour trend, vitals with high/low flags, lactate recency and stale-data markers for every bed." },
        { icon: Siren, t: "Alerts with accountability", d: "Review each alert with a reason. Reviews are shared with the team, so nobody re-checks the same alarm." },
        { icon: Clock3, t: "Look back in time", d: "Scrub through a stay hour by hour to see what the chart looked like when the risk changed." },
        { icon: ClipboardPlus, t: "New assessment", d: "Enter vitals and labs for any patient, at several time points, and get the same explained score." },
        { icon: ClipboardCheck, t: "Hour-1 bundle", d: "Surviving Sepsis Campaign reminders, pre-filled from the chart (e.g. 'last lactate 21 h ago')." },
        { icon: Users, t: "Access management", d: "Create patient sign-ins linked to one bed, reset passwords, and manage colleague accounts." },
      ],
    },
    patient: {
      title: "For patients",
      lead: "Your own record, explained without jargon — so you can follow what is happening and ask better questions.",
      items: [
        { icon: HeartPulse, t: "Your readings, explained", d: "Heart rate, blood pressure, breathing, temperature and oxygen, with 'in the usual range' or not." },
        { icon: Scale, t: "Risk as '12 in 100'", d: "A picture of 100 people shows your estimate, the clearest way to understand a chance." },
        { icon: Timer, t: "Looking ahead", d: "Whether your estimate for the next six hours is going up, down or staying the same." },
        { icon: BellRing, t: "Know you're being looked after", d: "If the tool raises an alert, you see that your care team was told — and when they reviewed it." },
        { icon: MessageCircleQuestion, t: "Questions to ask", d: "Warning signs to tell your nurse about straight away, and questions worth asking your team." },
        { icon: LockKeyhole, t: "Only your record", d: "Your sign-in is linked to you alone. You can never see another patient's information." },
      ],
    },
  }[role];

  return (
    <Section id="roles" className="border-y border-line bg-surface py-20 sm:py-28">
      <div data-reveal className="reveal flex flex-col items-center text-center">
        <Eyebrow>One record, two views</Eyebrow>
        <H2 className="max-w-2xl">Clinical depth for the team. Plain language for the patient.</H2>
        <div role="tablist" aria-label="Choose a view" className="mt-8 inline-flex rounded-xl border border-line bg-subtle p-1">
          {(
            [
              { id: "clinician", label: "I'm a clinician", icon: Stethoscope },
              { id: "patient", label: "I'm a patient", icon: HandHeart },
            ] as const
          ).map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={role === t.id}
              onClick={() => setRole(t.id)}
              className={clsx(
                "inline-flex h-10 items-center gap-2 rounded-lg px-4 text-sm font-semibold transition-colors",
                role === t.id ? "bg-surface text-ink shadow-card" : "text-muted hover:text-ink",
              )}
            >
              <t.icon className="size-4" />
              {t.label}
            </button>
          ))}
        </div>
      </div>
      <div key={role} className="landing-fade mt-12 grid items-start gap-10 lg:grid-cols-[1fr_1.6fr]">
        <div>
          <h3 className="text-2xl font-semibold tracking-tight text-ink">{content.title}</h3>
          <p className="mt-3 text-[15px] leading-7 text-muted">{content.lead}</p>
          <Link
            to="/login"
            className="mt-6 inline-flex h-11 items-center gap-2 rounded-xl bg-brand px-5 text-sm font-semibold text-white hover:bg-brand-hover"
          >
            Sign in as a {role} <ArrowRight className="size-4" />
          </Link>
          {role === "patient" && (
            <p className="mt-4 text-xs leading-5 text-faint">Your care team creates your account and gives you a temporary password.</p>
          )}
        </div>
        <ul className="grid gap-4 sm:grid-cols-2">
          {content.items.map((it) => (
            <li key={it.t} className="rounded-2xl border border-line bg-canvas p-5">
              <it.icon className={clsx("size-5", role === "clinician" ? "text-brand" : "text-ok")} />
              <h4 className="mt-3 text-[15px] font-semibold text-ink">{it.t}</h4>
              <p className="mt-1.5 text-sm leading-6 text-muted">{it.d}</p>
            </li>
          ))}
        </ul>
      </div>
    </Section>
  );
}

// ---------------------------------------------------------- principles

function Principles() {
  const items: { icon: LucideIcon; t: string; d: string; visual: ReactNode }[] = [
    {
      icon: EyeOff,
      t: "Never a false all-clear",
      d: "Low risk is shown in neutral grey, never green, and always says 'this does not rule out sepsis'.",
      visual: (
        <span className="inline-flex items-center gap-2">
          <span className="rounded-md bg-subtle px-2 py-0.5 text-sm font-semibold text-muted tnum">2%</span>
          <span className="text-xs text-muted">Not elevated — does not rule out sepsis</span>
        </span>
      ),
    },
    {
      icon: Clock3,
      t: "Old data looks old",
      d: "A normal heart rate from six hours ago isn't reassuring, so stale readings are greyed and timestamped.",
      visual: (
        <span className="inline-flex items-center gap-1.5 text-sm text-faint tnum">
          84 <Clock3 className="size-3.5" /> <span className="text-xs">last charted 6 h ago</span>
        </span>
      ),
    },
    {
      icon: BrainCircuit,
      t: "Every score explains itself",
      d: "No black box: each estimate lists the readings driving it, in the words clinicians use.",
      visual: (
        <span className="flex w-full flex-col gap-1">
          {[70, 45, 25].map((w, i) => (
            <span key={i} className="h-1.5 rounded-full bg-subtle">
              <span className="block h-full rounded-full bg-alert" style={{ width: `${w}%` }} />
            </span>
          ))}
        </span>
      ),
    },
    {
      icon: ShieldCheck,
      t: "Honest about its limits",
      d: "Real performance on unseen patients is published inside the app, including what it misses.",
      visual: (
        <span className="inline-flex items-center gap-2 text-xs text-muted">
          <CircleCheck className="size-4 text-ok" /> Model facts page, always one click away
        </span>
      ),
    },
  ];
  return (
    <Section className="py-20 sm:py-28">
      <div data-reveal className="reveal max-w-2xl">
        <Eyebrow>Design principles</Eyebrow>
        <H2>Designed so the right decision is the easy one.</H2>
        <p className="mt-4 text-lg leading-8 text-muted">
          Colour, wording and layout choices that stop a score from being over-trusted or ignored.
        </p>
      </div>
      <div className="mt-12 grid gap-5 sm:grid-cols-2">
        {items.map((p, i) => (
          <div key={p.t} data-reveal className="reveal flex flex-col rounded-2xl border border-line bg-surface p-6 shadow-card" style={{ transitionDelay: `${(i % 2) * 80}ms` }}>
            <div className="flex items-center gap-3">
              <span className="flex size-10 items-center justify-center rounded-xl bg-brand-soft text-brand-ink">
                <p.icon className="size-5" />
              </span>
              <h3 className="text-lg font-semibold text-ink">{p.t}</h3>
            </div>
            <p className="mt-3 text-[15px] leading-7 text-muted">{p.d}</p>
            <div className="mt-5 flex min-h-12 items-center rounded-xl border border-dashed border-line bg-canvas px-4 py-3">{p.visual}</div>
          </div>
        ))}
      </div>
    </Section>
  );
}

// ------------------------------------------------------------ evidence

function Evidence() {
  const { data } = usePublicSummary();
  const p = data?.pool;
  const metrics = [
    {
      v: p ? pctFmt(p.alert_precision) : "29%",
      k: "of alert hours fell within 6 hours of sepsis onset",
      note: "When it alerts, take it seriously.",
    },
    {
      v: p ? `${p.non_septic_patients_alerted} / ${(p.patients - p.septic_patients).toLocaleString()}` : "0 / 1,362",
      k: "patients who never developed sepsis received an alert",
      note: "Very few false alarms.",
    },
    {
      v: p ? `${p.septic_patients_alerted} / ${p.septic_patients}` : "9 / 138",
      k: "patients who developed sepsis were alerted",
      note: "Most cases are missed — a quiet tool is not reassurance.",
      warn: true,
    },
    {
      v: p ? p.hour_roc_auc.toFixed(2) : "0.68",
      k: "ROC-AUC on unseen stays (0.80 on the internal split)",
      note: "Moderate ability to rank risk.",
    },
  ];
  return (
    <Section id="evidence" className="border-y border-line bg-surface py-20 sm:py-28">
      <div className="grid gap-12 lg:grid-cols-[1fr_1.4fr]">
        <div data-reveal className="reveal">
          <Eyebrow>The evidence, without the spin</Eyebrow>
          <H2>Precise when it speaks. Quiet more often than it should be.</H2>
          <p className="mt-4 text-[15px] leading-7 text-muted">
            Tested on {p ? p.patients.toLocaleString() : "1,500"} real ICU stays from the public PhysioNet 2019 dataset that
            the model never saw during training. These figures are computed live by the server.
          </p>
          <div className="mt-6 rounded-2xl border border-warn-line bg-warn-soft p-5">
            <p className="text-sm leading-6 text-warn-ink">
              <strong className="font-semibold">How to use it:</strong> treat an alert as a strong prompt to review the
              patient. Never treat the absence of an alert as evidence that a patient does not have sepsis.
            </p>
          </div>
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          {metrics.map((m, i) => (
            <div
              key={m.k}
              data-reveal
              className={clsx("reveal rounded-2xl border p-6", m.warn ? "border-warn-line bg-canvas" : "border-line bg-canvas")}
              style={{ transitionDelay: `${(i % 2) * 80}ms` }}
            >
              <div className={clsx("text-4xl font-semibold tracking-tight tnum", m.warn ? "text-warn-ink" : "text-ink")}>{m.v}</div>
              <p className="mt-2 text-sm leading-6 text-muted">{m.k}</p>
              <p className={clsx("mt-4 border-t border-line pt-3 text-[13px] font-medium", m.warn ? "text-warn-ink" : "text-ink")}>{m.note}</p>
            </div>
          ))}
        </div>
      </div>
    </Section>
  );
}

// --------------------------------------------------------------- trust

function Trust() {
  const items: { icon: LucideIcon; t: string; d: string }[] = [
    { icon: LockKeyhole, t: "Role-based access", d: "Clinicians see the ward; each patient sees only their own record. Enforced on the server for every request." },
    { icon: ShieldCheck, t: "Secure sign-in", d: "Hashed passwords, signed sessions that expire, lockout after repeated failed attempts." },
    { icon: Database, t: "De-identified data", d: "Replays public, de-identified ICU stays from the PhysioNet/CinC 2019 Challenge. No real patients are tracked." },
    { icon: FlaskConical, t: "Research, not a device", d: "A research prototype for learning and evaluation. Not approved for clinical decisions." },
  ];
  return (
    <Section className="py-20 sm:py-28">
      <div data-reveal className="reveal mx-auto max-w-2xl text-center">
        <Eyebrow>Privacy & safety</Eyebrow>
        <H2>Built carefully, and clear about what it is.</H2>
      </div>
      <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {items.map((it, i) => (
          <div key={it.t} data-reveal className="reveal rounded-2xl border border-line bg-surface p-6 shadow-card" style={{ transitionDelay: `${i * 70}ms` }}>
            <it.icon className="size-6 text-brand" />
            <h3 className="mt-4 text-base font-semibold text-ink">{it.t}</h3>
            <p className="mt-2 text-sm leading-6 text-muted">{it.d}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}

// ----------------------------------------------------------------- FAQ

const FAQS: { q: string; a: string }[] = [
  {
    q: "Does a low score mean there is no sepsis?",
    a: "No. The model misses most cases of sepsis. A low score only means the tool has not found a warning pattern. Clinical assessment always comes first, and patients should tell their nurse if they feel worse.",
  },
  {
    q: "What happens when an alert fires?",
    a: "The patient moves to the top of the ward list. A clinician reviews the alert and records an outcome — for example, 'work-up started' or 'alternative explanation'. The patient sees that their team was alerted and when it was reviewed.",
  },
  {
    q: "What does '12 in 100' mean?",
    a: "Out of 100 patients with readings like yours, about 12 would be expected to develop sepsis in roughly the next six hours. It is an estimate for a group, not a prediction of exactly what will happen to one person.",
  },
  {
    q: "Who can see a patient's record?",
    a: "Clinicians with a SepsisSense account can see all patients on the ward. A patient's account is linked to their own record only. Patients do not see the clinical reasons or notes behind an alert review.",
  },
  {
    q: "What data does it use, and how often does it update?",
    a: "Vital signs (heart rate, blood pressure, breathing rate, temperature, oxygen) and blood tests (such as lactate, white cells and creatinine), plus age and time in the ICU. The estimate is recalculated every hour as new readings arrive.",
  },
  {
    q: "Is SepsisSense approved for clinical use?",
    a: "No. It is a research prototype built on public, de-identified data. It has not been validated prospectively or approved as a medical device, and must not be used to make decisions about real patients.",
  },
];

function Faq() {
  const [open, setOpen] = useState<number | null>(0);
  return (
    <Section id="faq" className="border-t border-line bg-surface py-20 sm:py-28">
      <div className="grid gap-10 lg:grid-cols-[1fr_1.6fr]">
        <div data-reveal className="reveal">
          <Eyebrow>Questions</Eyebrow>
          <H2>What people ask first.</H2>
          <p className="mt-4 text-[15px] leading-7 text-muted">
            Patients: if you are worried about how you feel right now, speak to your nurse — don't wait for a score.
          </p>
        </div>
        <ul className="divide-y divide-line rounded-2xl border border-line bg-canvas">
          {FAQS.map((f, i) => {
            const isOpen = open === i;
            return (
              <li key={f.q}>
                <h3>
                  <button
                    onClick={() => setOpen(isOpen ? null : i)}
                    aria-expanded={isOpen}
                    aria-controls={`faq-${i}`}
                    className="flex w-full items-center justify-between gap-4 px-6 py-5 text-left"
                  >
                    <span className="text-[15px] font-semibold text-ink">{f.q}</span>
                    <ChevronDown className={clsx("size-5 shrink-0 text-faint transition-transform", isOpen && "rotate-180")} />
                  </button>
                </h3>
                <div id={`faq-${i}`} hidden={!isOpen} className="px-6 pb-5 text-[15px] leading-7 text-muted">
                  {f.a}
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------ final CTA

function FinalCta() {
  return (
    <Section className="py-20 sm:py-24">
      <div data-reveal className="reveal relative overflow-hidden rounded-3xl bg-[#0f2a6b] px-8 py-14 text-center sm:px-16">
        <div className="pointer-events-none absolute -top-20 -left-20 size-72 rounded-full bg-white/5" aria-hidden />
        <div className="pointer-events-none absolute -right-16 -bottom-24 size-80 rounded-full bg-white/5" aria-hidden />
        <h2 className="relative text-3xl font-semibold tracking-tight text-balance text-white sm:text-4xl">Ready to take a look?</h2>
        <p className="relative mx-auto mt-4 max-w-xl text-[15px] leading-7 text-white/75">
          Sign in with the account your care team gave you. Exploring the research demo? Demo accounts are listed on the
          sign-in page.
        </p>
        <div className="relative mt-8 flex flex-wrap justify-center gap-3">
          <Link
            to="/login"
            className="inline-flex h-12 items-center gap-2 rounded-xl bg-white px-6 text-[15px] font-semibold text-[#0f2a6b] hover:bg-white/90"
          >
            Sign in <ArrowRight className="size-4" />
          </Link>
          <a
            href="#how"
            className="inline-flex h-12 items-center gap-2 rounded-xl border border-white/25 px-6 text-[15px] font-semibold text-white hover:bg-white/10"
          >
            How it works
          </a>
        </div>
      </div>
    </Section>
  );
}

// --------------------------------------------------------------- footer

function Footer() {
  const year = useRef(new Date().getFullYear()).current;
  return (
    <footer className="border-t border-line bg-surface px-5 py-10 sm:px-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-8 md:flex-row md:items-start md:justify-between">
        <div className="max-w-sm">
          <div className="flex items-center gap-2.5">
            <span className="flex size-8 items-center justify-center rounded-lg bg-brand text-white">
              <Activity className="size-[18px]" strokeWidth={2.4} />
            </span>
            <span className="text-[15px] font-semibold tracking-tight text-ink">SepsisSense</span>
          </div>
          <p className="mt-3 text-sm leading-6 text-muted">
            A research prototype for explainable sepsis early warning. Not a medical device; not for clinical use.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-10 text-sm sm:grid-cols-3">
          <div>
            <div className="font-semibold text-ink">Product</div>
            <ul className="mt-3 space-y-2 text-muted">
              {NAV.map((n) => (
                <li key={n.href}>
                  <a href={n.href} className="hover:text-ink">
                    {n.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="font-semibold text-ink">Account</div>
            <ul className="mt-3 space-y-2 text-muted">
              <li>
                <Link to="/login" className="hover:text-ink">
                  Sign in
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <div className="font-semibold text-ink">Project</div>
            <ul className="mt-3 space-y-2 text-muted">
              <li>
                <a href="https://github.com/suriyaprakash-25/SepsisSense" className="inline-flex items-center gap-1.5 hover:text-ink" target="_blank" rel="noreferrer">
                  <Github className="size-4" /> Source code
                </a>
              </li>
              <li>
                <a href="https://physionet.org/content/challenge-2019/" className="hover:text-ink" target="_blank" rel="noreferrer">
                  PhysioNet 2019 data
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>
      <div className="mx-auto mt-10 max-w-6xl border-t border-line pt-6 text-xs leading-5 text-faint">
        © {year} SepsisSense · MIT License · Data: Reyna MA et al., Early Prediction of Sepsis From Clinical Data: The
        PhysioNet/Computing in Cardiology Challenge 2019, Crit Care Med 2020;48(2):210–217.
      </div>
    </footer>
  );
}
