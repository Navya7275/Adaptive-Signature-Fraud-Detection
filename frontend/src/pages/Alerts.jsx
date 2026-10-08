import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Bell, CheckCircle2, AlertTriangle, ShieldAlert, Heart } from "lucide-react";
import {
  PageContainer,
  PageHeader,
  Card,
  EmptyState,
  Spinner,
} from "../components/Shared";
import { getAlerts, resolveAlert } from "../api/client";
import clsx from "clsx";

export default function Alerts() {
  const [alerts, setAlerts] = useState(null);
  const [filter, setFilter] = useState("active");
  const [resolvingId, setResolvingId] = useState(null);

  const load = async () => {
    try {
      const data = await getAlerts(filter === "resolved");
      setAlerts(data);
    } catch {
      setAlerts([]);
    }
  };

  useEffect(() => {
    setAlerts(null);
    load();
  }, [filter]);

  const handleResolve = async (id) => {
    setResolvingId(id);
    try {
      await resolveAlert(id);
      await load();
    } finally {
      setResolvingId(null);
    }
  };

  const typeConfig = {
    forgery_suspected: {
      icon: ShieldAlert,
      label: "Forgery Suspected",
      color: "text-danger bg-danger/10 border-danger/20",
    },
    medical_event: {
      icon: Heart,
      label: "Medical Event",
      color: "text-warning bg-warning/10 border-warning/20",
    },
    rapid_drift: {
      icon: AlertTriangle,
      label: "Rapid Drift",
      color: "text-info bg-info/10 border-info/20",
    },
  };

  return (
    <PageContainer>
      <PageHeader
        title="Alerts"
        subtitle="Flagged events that require attention"
        actions={
          <div className="flex items-center gap-1 p-1 rounded-full bg-neutral-100 dark:bg-surface-elevated">
            {["active", "resolved"].map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={clsx(
                  "px-3 py-1 text-xs font-medium rounded-full capitalize transition-all",
                  filter === f
                    ? "bg-white dark:bg-surface-card shadow-sm"
                    : "text-neutral-500 hover:text-neutral-900 dark:hover:text-neutral-100"
                )}
              >
                {f}
              </button>
            ))}
          </div>
        }
      />

      {alerts === null ? (
        <div className="py-20 text-center">
          <Spinner />
        </div>
      ) : alerts.length === 0 ? (
        <EmptyState
          icon={filter === "active" ? CheckCircle2 : Bell}
          title={filter === "active" ? "All clear" : "No resolved alerts"}
          description={
            filter === "active"
              ? "No alerts need your attention right now."
              : "Resolved alerts will appear here."
          }
        />
      ) : (
        <div className="space-y-3">
          {alerts.map((a, i) => {
            const config = typeConfig[a.alert_type] || typeConfig.rapid_drift;
            const Icon = config.icon;
            return (
              <motion.div
                key={a.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04 }}
              >
                <Card hoverable={false}>
                  <div className="flex items-start gap-4">
                    <div
                      className={clsx(
                        "w-11 h-11 rounded-xl flex items-center justify-center border flex-shrink-0",
                        config.color
                      )}
                    >
                      <Icon className="w-5 h-5" />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap mb-1">
                        <h3 className="font-semibold">{a.user_name}</h3>
                        <span
                          className={clsx(
                            "text-[10px] px-1.5 py-0.5 rounded font-semibold uppercase",
                            a.severity === "critical"
                              ? "bg-danger/10 text-danger"
                              : a.severity === "high"
                              ? "bg-warning/10 text-warning"
                              : "bg-info/10 text-info"
                          )}
                        >
                          {a.severity}
                        </span>
                        <span
                          className={clsx(
                            "text-[10px] px-1.5 py-0.5 rounded font-medium",
                            config.color
                          )}
                        >
                          {config.label}
                        </span>
                      </div>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-2">
                        {a.description}
                      </p>
                      <p className="text-[10px] text-neutral-500 dark:text-neutral-500">
                        {new Date(a.created_at).toLocaleString()}
                      </p>
                    </div>

                    {!a.resolved && (
                      <button
                        onClick={() => handleResolve(a.id)}
                        disabled={resolvingId === a.id}
                        className="btn-secondary text-xs flex-shrink-0"
                      >
                        {resolvingId === a.id ? <Spinner size="sm" /> : "Resolve"}
                      </button>
                    )}
                  </div>
                </Card>
              </motion.div>
            );
          })}
        </div>
      )}
    </PageContainer>
  );
}