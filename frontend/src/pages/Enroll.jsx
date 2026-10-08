import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { UserPlus, CheckCircle2 } from "lucide-react";
import {
  PageContainer,
  PageHeader,
  Card,
  Spinner,
} from "../components/Shared";
import SignatureUpload from "../components/SignatureUpload";
import { enrollUser } from "../api/client";

export default function Enroll() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(null);
  const [error, setError] = useState("");

  const canSubmit = name.trim() && age && files.length >= 1;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!canSubmit || loading) return;
    setError("");
    setLoading(true);
    try {
      const result = await enrollUser(name.trim(), parseInt(age), files);
      setSuccess(result);
      setTimeout(() => navigate(`/users/${result.user_id}`), 1600);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Enrollment failed");
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <PageContainer>
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="max-w-md mx-auto text-center py-20"
        >
          <div className="inline-flex w-16 h-16 rounded-2xl bg-success/10 items-center justify-center mb-4">
            <CheckCircle2 className="w-8 h-8 text-success" />
          </div>
          <h2 className="text-2xl font-bold">Enrollment successful</h2>
          <p className="mt-2 text-sm text-neutral-500 dark:text-neutral-400">
            {success.name} has been enrolled with {success.reference_signatures} reference
            signatures.
          </p>
          <p className="mt-6 text-xs text-neutral-400">Redirecting to profile...</p>
        </motion.div>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <PageHeader
        title="Enroll New User"
        subtitle="Capture baseline signatures to begin drift profiling"
      />

      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left — user info */}
          <Card hoverable={false} className="lg:col-span-1">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <UserPlus className="w-4 h-4 text-accent" />
              User Details
            </h3>

            <div className="space-y-4">
              <div>
                <label className="text-xs font-medium text-neutral-600 dark:text-neutral-400 block mb-1.5">
                  Full Name
                </label>
                <input
                  type="text"
                  className="input"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Anjali Sharma"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-neutral-600 dark:text-neutral-400 block mb-1.5">
                  Age
                </label>
                <input
                  type="number"
                  className="input"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  placeholder="e.g. 45"
                  min="1"
                  max="120"
                />
              </div>
            </div>

            {error && (
              <div className="mt-4 p-3 rounded-xl bg-danger/10 border border-danger/20 text-xs text-danger">
                {error}
              </div>
            )}
          </Card>

          {/* Right — signatures */}
          <Card hoverable={false} className="lg:col-span-2">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-semibold">Reference Signatures</h3>
                <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-0.5">
                  Upload 3–5 genuine signatures for best results
                </p>
              </div>
              <span className="text-xs font-medium text-accent">
                {files.length}/5
              </span>
            </div>
            <SignatureUpload multiple maxFiles={5} onFilesChange={setFiles} />
          </Card>
        </div>

        <div className="mt-6 flex justify-end gap-3">
          <button
            type="button"
            onClick={() => navigate(-1)}
            className="btn-ghost"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!canSubmit || loading}
            className="btn-primary"
          >
            {loading ? (
              <>
                <Spinner size="sm" /> Enrolling...
              </>
            ) : (
              <>Enroll User</>
            )}
          </button>
        </div>
      </form>
    </PageContainer>
  );
}