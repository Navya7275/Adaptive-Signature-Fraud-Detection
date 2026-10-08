import { Sun, Moon } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useTheme } from "../context/ThemeContext";

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === "dark";

  return (
    <button
      onClick={toggleTheme}
      aria-label="Toggle theme"
      className="glass rounded-full w-10 h-10 flex items-center justify-center shadow-float hover:scale-105 transition-transform"
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={theme}
          initial={{ opacity: 0, rotate: -90, scale: 0.5 }}
          animate={{ opacity: 1, rotate: 0, scale: 1 }}
          exit={{ opacity: 0, rotate: 90, scale: 0.5 }}
          transition={{ duration: 0.2 }}
        >
          {isDark ? (
            <Moon className="w-4 h-4 text-neutral-300" />
          ) : (
            <Sun className="w-4 h-4 text-amber-500" />
          )}
        </motion.div>
      </AnimatePresence>
    </button>
  );
}