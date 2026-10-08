import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, User, Calendar, Activity, Clock } from "lucide-react";
import {
  PageContainer,
  Card,
  DecisionBadge,
  DriftBadge,
  DriftDirection,
  StatCard,
  Skeleton,
  EmptyState,
} from "../components/Shared";
import DriftChart from "../components/DriftChart";
import { getUser, getUserHistory } from "../api/client";

export default function UserDetail() {
  const { userId } = useParams();
  const [user, setUser] = useState(null);
  const [history, setHistory] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const [u, h] = await Promise.all([getUser(userId), getUserHistory(userId)]);
        setUser(u);
        setHistory(h);
      } catch {
        setUser(false);
      }
    })();
  }, [userId]);

  if (user === null) {
    return (
      <PageContainer>
        <Skeleton className="h-12 w-64 mb-6" />
        <div className="grid grid-cols-4 gap-4 mb-6">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
        <Skeleton className="h-80" />
      </PageContainer>
    );
  }

  if (user === false) {
    return (
      <PageContainer>
        <EmptyState
          icon={User}
          title="User not found"
          description="This user does not exist or has been removed."
          action={
            <Link to="/" className="btn-primary">
              Back to dashboard
            </Link>
          }
        />
      </PageContainer>
    );
  }

  const profile = user.drift_profile || {};
  const signatures = history?.signatures || [];
  const logs = history?.verification_logs || [];

  return (
    <PageContainer>
      <Link to="/" className="btn-ghost mb-4 -ml-3">
        <ArrowLeft className="w-4 h-4" /> Back to dashboard
      </Link>

      {/* User header */}
      <div className="flex items-start justify-between mb-8">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-accent-light to-accent flex items-center justify-center text-white font-bold text-xl shadow-glow">
            {user.name?.charAt(0).toUpperCase()}
          </div>
          <div>
            <h1 className="text-2xl font-bold">{user.name}</h1>
            <div className="flex items-center gap-3 mt-1 text-xs text-neutral-500 dark:text-neutral-400">
              <span className="flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                Enrolled {user.enrollment_date}
              </span>
              <span>Age {user.age_at_enrollment}</span>
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded uppercase font-semibold ${
                  user.status === "active"
                    ? "bg-success/10 text-success"
                    : "bg-neutral-500/10 text-neutral-500"
                }`}
              >
                {user.status}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Drift profile stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatCard
          label="Tenure"
          value={`${profile.tenure_months || 0} mo`}
          icon={Clock}
          color="info"
        />
        <StatCard
          label="Drift Rate / mo"
          value={(profile.drift_rate || 0).toFixed(4)}
          icon={Activity}
          color={profile.drift_rate < 0 ? "warning" : "success"}
        />
        <StatCard
          label="Volatility"
          value={(profile.volatility || 0).toFixed(3)}
          icon={Activity}
          color="accent"
        />
        <StatCard
          label="Trust Buffer"
          value={`+${(profile.trust_buffer || 0).toFixed(3)}`}
          icon={Activity}
          color="success"
        />
      </div>

      {/* Drift chart */}
      <Card hoverable={false} className="mb-6">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="text-lg font-semibold">Signature Drift Timeline</h2>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
              Similarity score over time with adaptive threshold
            </p>
          </div>
          <div className="flex items-center gap-2">
            <DriftDirection direction={profile.drift_direction || "stable"} />
          </div>
        </div>
        {signatures.length > 0 ? (
          <DriftChart
            signatures={signatures}
            threshold={(profile.base_threshold || 0.8) - (profile.trust_buffer || 0)}
          />
        ) : (
          <EmptyState
            icon={Activity}
            title="No drift data yet"
            description="Submit more verifications over time to see the drift pattern."
          />
        )}
      </Card>

      {/* Verification log */}
      <Card hoverable={false}>
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="text-lg font-semibold">Verification History</h2>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
              Last {logs.length} verification{logs.length !== 1 ? "s" : ""}
            </p>
          </div>
        </div>
        {logs.length === 0 ? (
          <EmptyState
            icon={Activity}
            title="No verifications yet"
            description="Run a verification to see logs appear here."
          />
        ) : (
          <div className="space-y-2">
            {logs
              .slice()
              .reverse()
              .slice(0, 10)
              .map((log, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.03 }}
                  className="p-4 rounded-xl border border-neutral-200 dark:border-surface-border"
                >
                  <div className="flex items-start justify-between mb-2 gap-3">
                    <div className="flex items-center gap-2 flex-wrap">
                      <DecisionBadge decision={log.decision} size="sm" />
                      {log.drift_classification && (
                        <DriftBadge
                          classification={log.drift_classification}
                          size="sm"
                        />
                      )}
                    </div>
                    <span className="text-[10px] text-neutral-500 dark:text-neutral-400 whitespace-nowrap">
                      {new Date(log.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-xs mb-2">
                    <span>
                      Score:{" "}
                      <span className="font-mono font-semibold text-accent">
                        {log.raw_score?.toFixed(3)}
                      </span>
                    </span>
                    <span className="text-neutral-500 dark:text-neutral-400">
                      Threshold: {log.threshold_used?.toFixed(3)}
                    </span>
                    <span className="text-neutral-500 dark:text-neutral-400">
                      Confidence: {(log.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  {log.reason && (
                    <p className="text-xs text-neutral-600 dark:text-neutral-400 line-clamp-2">
                      {log.reason}
                    </p>
                  )}
                </motion.div>
              ))}
          </div>
        )}
      </Card>
    </PageContainer>
  );
}