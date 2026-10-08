import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, X, FileImage } from "lucide-react";
import clsx from "clsx";

export default function SignatureUpload({
  multiple = false,
  onFilesChange,
  maxFiles = 5,
}) {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef();

  // Create preview URLs and revoke them when files change/unmount,
  // so each preview doesn't leak an object URL for the page lifetime.
  useEffect(() => {
    const urls = files.map((f) => URL.createObjectURL(f));
    setPreviews(urls);
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
  }, [files]);

  const handleFiles = (newFiles) => {
    const arr = Array.from(newFiles).filter((f) => f.type.startsWith("image/"));
    let updated;
    if (multiple) {
      updated = [...files, ...arr].slice(0, maxFiles);
    } else {
      updated = arr.slice(0, 1);
    }
    setFiles(updated);
    onFilesChange?.(updated);
  };

  const removeFile = (idx) => {
    const updated = files.filter((_, i) => i !== idx);
    setFiles(updated);
    onFilesChange?.(updated);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
  };

  return (
    <div>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={clsx(
          "relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all",
          dragActive
            ? "border-accent bg-accent/5 scale-[1.01]"
            : "border-neutral-300 dark:border-surface-border hover:border-accent/50 hover:bg-neutral-50 dark:hover:bg-surface-elevated/50"
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple={multiple}
          onChange={(e) => handleFiles(e.target.files)}
          className="hidden"
        />
        <motion.div
          animate={{ y: dragActive ? -4 : 0 }}
          className="inline-flex w-14 h-14 rounded-2xl bg-accent/10 items-center justify-center mb-4"
        >
          <Upload className="w-6 h-6 text-accent" />
        </motion.div>
        <p className="font-semibold">
          {dragActive ? "Drop to upload" : "Drop signature image or click to browse"}
        </p>
        <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
          PNG, JPG up to 10MB {multiple && `• Max ${maxFiles} files`}
        </p>
      </div>

      {/* Preview */}
      <AnimatePresence>
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3"
          >
            {files.map((file, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="relative group rounded-xl overflow-hidden border border-neutral-200 dark:border-surface-border bg-white dark:bg-surface-elevated"
              >
                <div className="aspect-[3/2] flex items-center justify-center p-2">
                  <img
                    src={previews[i]}
                    alt={file.name}
                    className="max-w-full max-h-full object-contain"
                  />
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(i);
                  }}
                  className="absolute top-1.5 right-1.5 w-6 h-6 rounded-full bg-black/60 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
                <p className="text-[10px] p-2 truncate border-t border-neutral-200 dark:border-surface-border text-neutral-600 dark:text-neutral-400">
                  <FileImage className="inline w-3 h-3 mr-1" />
                  {file.name}
                </p>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}