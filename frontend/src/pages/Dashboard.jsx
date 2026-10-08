import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Users,
  ShieldCheck,
  AlertTriangle,
  Activity,
  ArrowRight,
  Calendar,
} from "lucide-react";
import {
  PageContainer,
  PageHeader,
  StatCard,
  Card,
  DriftBadge,
  DriftDirection,
  EmptyState,
  Skeleton,
} from "../components/Shared";
import { listUsers, getAlerts } from "../api/client";

export default function Dashboard() {
  const [users, setUsers] = useState(null);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        const [u, a] = await Promise.all([listUsers(), getAlerts(false)]);
        setUsers(u);
        setAlerts(a);
      } catch {
        setUsers([]);
        setAlerts([]);
      }
    })();
  }, []);

  const totalUsers = users?.length ?? 0;
  const totalVerifications = users?.reduce((acc, u) => acc + (u.sig_count || 0), 0) ?? 0;
  const activeAlerts = alerts.length;

  return (
    <PageContainer>
      <PageHeader
        title="Dashboard"
        subtitle="Overview of your adaptive signature verification system"
      />

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard label="Enrolled Users" value={totalUsers} icon={Users} color="accent" />
        <StatCard
          label="Total Signatures"
          value={totalVerifications}
          icon={ShieldCheck}
          color="info"
        />
        <StatCard
          label="Active Alerts"
          value={activeAlerts}
          icon={AlertTriangle}
          color={activeAlerts > 0 ? "warning" : "success"}
        />
        <StatCard
          label="System Status"
          value="Online"
          icon={Activity}
          color="success"
          trend="All services healthy"
        />
      </div>

      {/* Users list */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card hoverable={false}>
            <div className="flex items-center justify-between mb-5">
              <div>
                <h2 className="text-lg font-semibold">Enrolled Users</h2>
                <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                  Active drift profiles
                </p>
              </div>
              <Link
                to="/enroll"
                className="btn-secondary text-xs"
              >
                + Enroll new
              </Link>
            </div>

            {users === null ? (
              <div className="space-y-3">
                {[0, 1, 2].map((i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : users.length === 0 ? (
              <EmptyState
                icon={Users}
                title="No users yet"
                description="Enroll your first user to start tracking signatures."
                action={
                  <Link to="/enroll" className="btn-primary">
                    Enroll first user
                  </Link>
                }
              />
            ) : (
              <div className="space-y-2">
                {users.map((u, i) => (
                  <motion.div
                    key={u.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.04 }}
                  >
                    <Link
                      to={`/users/${u.id}`}
                      className="flex items-center justify-between p-4 rounded-xl border border-neutral-200 dark:border-surface-border hover:border-accent/30 hover:bg-neutral-50 dark:hover:bg-surface-elevated transition-all group"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-accent-light to-accent flex items-center justify-center text-white font-bold text-sm">
                          {u.name?.charAt(0).toUpperCase() || "?"}
                        </div>
                        <div>
                          <p className="font-medium text-sm">{u.name}</p>
                          <div className="flex items-center gap-3 mt-0.5 text-xs text-neutral-500 dark:text-neutral-400">
                            <span className="flex items-center gap-1">
                              <Calendar className="w-3 h-3" />
                              {u.tenure_months || 0}mo
                            </span>
                            <span>{u.sig_count || 0} signatures</span>
                            <DriftDirection direction={u.drift_direction || "stable"} />
                          </div>
                        </div>
                      </div>
                      <ArrowRight className="w-4 h-4 text-neutral-400 group-hover:text-accent group-hover:translate-x-0.5 transition-all" />
                    </Link>
                  </motion.div>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* Recent alerts */}
        <Card hoverable={false}>
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-semibold">Recent Alerts</h2>
            <Link to="/alerts" className="text-xs text-accent hover:underline">
              View all
            </Link>
          </div>

          {alerts.length === 0 ? (
            <div className="text-center py-8">
              <div className="inline-flex w-10 h-10 rounded-full bg-success/10 items-center justify-center mb-2">
                <ShieldCheck className="w-5 h-5 text-success" />
              </div>
              <p className="text-sm font-medium">All clear</p>
              <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                No active alerts
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {alerts.slice(0, 5).map((a) => (
                <div
                  key={a.id}
                  className="p-3 rounded-xl border border-neutral-200 dark:border-surface-border"
                >
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <p className="text-xs font-medium">{a.user_name}</p>
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded font-semibold uppercase ${
                        a.severity === "critical"
                          ? "bg-danger/10 text-danger"
                          : "bg-warning/10 text-warning"
                      }`}
                    >
                      {a.severity}
                    </span>
                  </div>
                  <p className="text-xs text-neutral-500 dark:text-neutral-400 line-clamp-2">
                    {a.description}
                  </p>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </PageContainer>
  );
}