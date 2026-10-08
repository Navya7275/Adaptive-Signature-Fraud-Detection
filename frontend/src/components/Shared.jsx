import { motion } from "framer-motion";
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  TrendingDown,
  TrendingUp,
  Minus,
  Activity,
  Heart,
  ShieldAlert,
  Circle,
} from "lucide-react";
import clsx from "clsx";

// ────────────────────────────────────────
//  PAGE LAYOUT
// ────────────────────────────────────────
export function PageContainer({ children, className }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={clsx("max-w-6xl mx-auto px-6 pt-28 pb-20", className)}
    >
      {children}
    </motion.div>
  );
}

export function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="flex items-start justify-between mb-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">{title}</h1>
        {subtitle && (
          <p className="mt-1.5 text-sm text-neutral-500 dark:text-neutral-400">
            {subtitle}
          </p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

// ────────────────────────────────────────
//  CARD
// ────────────────────────────────────────
export function Card({ children, className, hoverable = true, ...props }) {
  return (
    <div
      className={clsx(
        "card-float p-6",
        hoverable ? "" : "hover:translate-y-0 hover:shadow-float",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

// ────────────────────────────────────────
//  STAT CARD
// ────────────────────────────────────────
export function StatCard({ label, value, icon: Icon, trend, color = "accent" }) {
  const colors = {
    accent: "text-accent bg-accent/10",
    success: "text-success bg-success/10",
    warning: "text-warning bg-warning/10",
    danger: "text-danger bg-danger/10",
    info: "text-info bg-info/10",
  };

  return (
    <Card className="group">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
            {label}
          </p>
          <p className="mt-2 text-3xl font-bold tracking-tight">{value}</p>
          {trend && (
            <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
              {trend}
            </p>
          )}
        </div>
        {Icon && (
          <div
            className={clsx(
              "w-11 h-11 rounded-xl flex items-center justify-center transition-transform group-hover:scale-110",
              colors[color]
            )}
          >
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>
    </Card>
  );
}

// ────────────────────────────────────────
//  DECISION BADGE
// ────────────────────────────────────────
export function DecisionBadge({ decision, size = "md" }) {
  const configs = {
    approved: {
      label: "Approved",
      icon: CheckCircle2,
      className: "bg-success/10 text-success border-success/20",
    },
    escalated: {
      label: "Escalated",
      icon: AlertTriangle,
      className: "bg-warning/10 text-warning border-warning/20",
    },
    rejected: {
      label: "Rejected",
      icon: XCircle,
      className: "bg-danger/10 text-danger border-danger/20",
    },
  };

  const config = configs[decision] || configs.escalated;
  const Icon = config.icon;

  const sizes = {
    sm: "text-xs px-2 py-0.5",
    md: "text-sm px-3 py-1",
    lg: "text-base px-4 py-1.5",
  };

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full font-medium border",
        config.className,
        sizes[size]
      )}
    >
      <Icon className={clsx(size === "sm" ? "w-3 h-3" : "w-4 h-4")} />
      {config.label}
    </span>
  );
}

// ────────────────────────────────────────
//  DRIFT CLASSIFICATION BADGE
// ────────────────────────────────────────
export function DriftBadge({ classification, size = "md" }) {
  const configs = {
    normal: {
      label: "Normal",
      icon: Circle,
      className: "bg-neutral-500/10 text-neutral-500 border-neutral-500/20",
    },
    natural_aging: {
      label: "Natural Aging",
      icon: TrendingDown,
      className: "bg-info/10 text-info border-info/20",
    },
    medical_event: {
      label: "Medical Event",
      icon: Heart,
      className: "bg-warning/10 text-warning border-warning/20",
    },
    forgery_attempt: {
      label: "Forgery Attempt",
      icon: ShieldAlert,
      className: "bg-danger/10 text-danger border-danger/20",
    },
  };

  const config = configs[classification] || configs.normal;
  const Icon = config.icon;

  const sizes = {
    sm: "text-xs px-2 py-0.5",
    md: "text-sm px-3 py-1",
  };

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full font-medium border",
        config.className,
        sizes[size]
      )}
    >
      <Icon className={clsx(size === "sm" ? "w-3 h-3" : "w-3.5 h-3.5")} />
      {config.label}
    </span>
  );
}

// ────────────────────────────────────────
//  DRIFT DIRECTION ICON
// ────────────────────────────────────────
export function DriftDirection({ direction }) {
  const configs = {
    degrading: { icon: TrendingDown, color: "text-warning" },
    improving: { icon: TrendingUp, color: "text-danger" },
    stable: { icon: Minus, color: "text-success" },
  };
  const config = configs[direction] || configs.stable;
  const Icon = config.icon;
  return (
    <span className={clsx("inline-flex items-center gap-1", config.color)}>
      <Icon className="w-4 h-4" />
      <span className="text-xs capitalize">{direction}</span>
    </span>
  );
}

// ────────────────────────────────────────
//  EMPTY STATE
// ────────────────────────────────────────
export function EmptyState({ icon: Icon = Activity, title, description, action }) {
  return (
    <div className="text-center py-20">
      <div className="inline-flex w-14 h-14 rounded-2xl bg-neutral-100 dark:bg-surface-elevated items-center justify-center mb-4">
        <Icon className="w-6 h-6 text-neutral-400" />
      </div>
      <h3 className="text-lg font-semibold">{title}</h3>
      {description && (
        <p className="mt-1 text-sm text-neutral-500 dark:text-neutral-400 max-w-sm mx-auto">
          {description}
        </p>
      )}
      {action && <div className="mt-6">{action}</div>}
    </div>
  );
}

// ────────────────────────────────────────
//  SKELETON
// ────────────────────────────────────────
export function Skeleton({ className }) {
  return (
    <div
      className={clsx(
        "animate-pulse bg-neutral-200 dark:bg-surface-elevated rounded-lg",
        className
      )}
    />
  );
}

// ────────────────────────────────────────
//  SPINNER
// ────────────────────────────────────────
export function Spinner({ size = "md" }) {
  const sizes = { sm: "w-4 h-4", md: "w-6 h-6", lg: "w-8 h-8" };
  return (
    <svg
      className={clsx("animate-spin text-accent", sizes[size])}
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeDasharray="31.4 31.4"
        strokeDashoffset="0"
      />
    </svg>
  );
}