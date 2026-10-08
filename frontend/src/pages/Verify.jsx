import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  ShieldCheck,
  FileText,
  ArrowRight,
  Sparkles,
  Info,
} from "lucide-react";
import {
  PageContainer,
  PageHeader,
  Card,
  DecisionBadge,
  DriftBadge,
  Spinner,
} from "../components/Shared";
import SignatureUpload from "../components/SignatureUpload";
import { listUsers, verifySignature } from "../api/client";

export default function Verify() {
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  // Remount key for SignatureUpload: cleared after every verification so
  // a stale file can never be silently re-submitted.
  const [uploadKey, setUploadKey] = useState(0);

  useEffect(() => {
    listUsers().then(setUsers).catch(() => setUsers([]));
  }, []);

  const selectedUser = users.find((u) => u.id === selectedUserId);

  const handleVerify = async () => {
    if (!selectedUserId || files.length === 0 || loading) return;
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const r = await verifySignature(selectedUserId, files[0]);
      setResult(r);
      // Clear the upload box — the next verification must get a fresh file
      setFiles([]);
      setUploadKey((k) => k + 1);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Verification failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageContainer>
      <PageHeader
        title="Verify Signature"
        subtitle="Run adaptive verification against a user's drift profile"
      />

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left — Input */}
        <div className="lg:col-span-2 space-y-6">
          <Card hoverable={false}>
            <h3 className="font-semibold mb-4 text-sm">1. Select User</h3>
            <select
              className="input"
              value={selectedUserId}
              onChange={(e) => setSelectedUserId(e.target.value)}
            >
              <option value="">Choose a user...</option>
              {users.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.name} ({u.tenure_months || 0}mo tenure)
                </option>
              ))}
            </select>
            {selectedUser && (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-3 p-3 rounded-xl bg-accent/5 border border-accent/20 text-xs"
              >
                <p className="font-semibold">{selectedUser.name}</p>
                <p className="text-neutral-600 dark:text-neutral-400 mt-0.5">
                  Age {selectedUser.age_at_enrollment} • {selectedUser.sig_count} signatures •{" "}
                  {selectedUser.drift_direction || "stable"}
                </p>
              </motion.div>
            )}
          </Card>

          <Card hoverable={false}>
            <h3 className="font-semibold mb-4 text-sm">2. Upload Signature</h3>
            <SignatureUpload key={uploadKey} onFilesChange={setFiles} />
          </Card>

          <button
            onClick={handleVerify}
            disabled={!selectedUserId || files.length === 0 || loading}
            className="btn-primary w-full py-3 text-base"
          >
            {loading ? (
              <>
                <Spinner size="sm" /> Analyzing signature...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" /> Run Verification
              </>
            )}
          </button>

          {error && (
            <div className="p-3 rounded-xl bg-danger/10 border border-danger/20 text-xs text-danger">
              {error}
            </div>
          )}
        </div>

        {/* Right — Result */}
        <div className="lg:col-span-3">
          <AnimatePresence mode="wait">
            {!result && !loading && (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <Card hoverable={false} className="min-h-[500px] flex items-center justify-center">
                  <div className="text-center">
                    <div className="inline-flex w-14 h-14 rounded-2xl bg-neutral-100 dark:bg-surface-elevated items-center justify-center mb-4">
                      <ShieldCheck className="w-6 h-6 text-neutral-400" />
                    </div>
                    <p className="font-semibold">Ready to verify</p>
                    <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1 max-w-xs">
                      Select a user and upload a signature to begin adaptive drift-aware
                      verification
                    </p>
                  </div>
                </Card>
              </motion.div>
            )}

            {loading && (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <Card hoverable={false} className="min-h-[500px] flex items-center justify-center">
                  <div className="text-center">
                    <Spinner size="lg" />
                    <p className="mt-4 font-medium">Analyzing signature</p>
                    <div className="mt-2 text-xs text-neutral-500 dark:text-neutral-400 space-y-0.5">
                      <p>Extracting 128-dim embedding...</p>
                      <p>Computing stroke features...</p>
                      <p>Running drift analysis...</p>
                    </div>
                  </div>
                </Card>
              </motion.div>
            )}

            {result && (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
              >
                <ResultCard result={result} user={selectedUser} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </PageContainer>
  );
}

function ResultCard({ result, user }) {
  const decisionColors = {
    approved: "from-success/20 to-success/5",
    escalated: "from-warning/20 to-warning/5",
    rejected: "from-danger/20 to-danger/5",
  };

  return (
    <div className="space-y-4">
      {/* Main decision card */}
      <div
        className={`card-float p-8 bg-gradient-to-br ${
          decisionColors[result.decision]
        }`}
      >
        <div className="flex items-start justify-between mb-6">
          <div>
            <p className="text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider mb-2">
              Verification Result
            </p>
            <DecisionBadge decision={result.decision} size="lg" />
          </div>
          <DriftBadge classification={result.drift_classification} />
        </div>

        {/* Score meters */}
        <div className="grid grid-cols-3 gap-3 mb-6">
          <ScoreDisplay
            label="Similarity"
            value={result.raw_score}
            highlight
          />
          <ScoreDisplay label="Threshold" value={result.threshold_used} />
          <ScoreDisplay label="Confidence" value={result.confidence} />
        </div>

        {/* Reason */}
        <div className="p-4 rounded-xl bg-white/60 dark:bg-surface-card/60 backdrop-blur">
          <div className="flex items-start gap-2">
            <Info className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
            <p className="text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
              {result.reason}
            </p>
          </div>
        </div>
      </div>

      {/* Details */}
      <div className="grid grid-cols-2 gap-3">
        <DetailCard label="Tenure" value={`${result.tenure_months} months`} />
        <DetailCard
          label="Trust Buffer"
          value={`+${result.trust_buffer.toFixed(3)}`}
        />
        <DetailCard
          label="Drift Direction"
          value={result.drift_direction}
          capitalize
        />
        <DetailCard
          label="Samples Analyzed"
          value={result.drift_details?.num_samples || 1}
        />
      </div>

      {user && (
        <Link
          to={`/users/${user.id}`}
          className="btn-secondary w-full justify-between"
        >
          <span className="flex items-center gap-2">
            <FileText className="w-4 h-4" /> View full drift timeline
          </span>
          <ArrowRight className="w-4 h-4" />
        </Link>
      )}
    </div>
  );
}

function ScoreDisplay({ label, value, highlight }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="text-center">
      <p className="text-xs text-neutral-500 dark:text-neutral-400 mb-1">{label}</p>
      <p
        className={`text-2xl font-bold tabular-nums ${
          highlight ? "text-accent" : ""
        }`}
      >
        {(value || 0).toFixed(3)}
      </p>
      <div className="mt-2 h-1 rounded-full bg-neutral-200 dark:bg-surface-elevated overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className={`h-full ${highlight ? "bg-accent" : "bg-neutral-400"}`}
        />
      </div>
    </div>
  );
}

function DetailCard({ label, value, capitalize }) {
  return (
    <div className="card-float p-4">
      <p className="text-[10px] font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
        {label}
      </p>
      <p className={`mt-1 text-lg font-semibold ${capitalize ? "capitalize" : ""}`}>
        {value}
      </p>
    </div>
  );
}