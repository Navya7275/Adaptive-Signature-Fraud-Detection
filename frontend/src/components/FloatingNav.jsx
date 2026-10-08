import { NavLink, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { LayoutDashboard, UserPlus, ShieldCheck, Bell, Fingerprint } from "lucide-react";
import ThemeToggle from "./ThemeToggle";
import clsx from "clsx";

const items = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/enroll", label: "Enroll", icon: UserPlus },
  { to: "/verify", label: "Verify", icon: ShieldCheck },
  { to: "/alerts", label: "Alerts", icon: Bell },
];

export default function FloatingNav() {
  const location = useLocation();

  return (
    <>
      {/* Brand — top left, floating */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="fixed top-6 left-6 z-50 flex items-center gap-2"
      >
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent-light to-accent flex items-center justify-center shadow-glow">
          <Fingerprint className="w-5 h-5 text-white" />
        </div>
        <div className="hidden sm:block">
          <p className="text-sm font-bold tracking-tight">SignaDrift</p>
          <p className="text-[10px] text-neutral-500 -mt-0.5">
            Adaptive Verification
          </p>
        </div>
      </motion.div>

      {/* Center floating pill nav */}
      <motion.nav
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.05 }}
        className="fixed top-6 left-1/2 -translate-x-1/2 z-50 glass rounded-full px-2 py-1.5 shadow-float"
      >
        <div className="flex items-center gap-0.5">
          {items.map(({ to, label, icon: Icon }) => {
            const active =
              to === "/"
                ? location.pathname === "/"
                : location.pathname.startsWith(to);
            return (
              <NavLink
                key={to}
                to={to}
                className={clsx(
                  "relative flex items-center gap-2 px-3.5 py-2 rounded-full text-sm font-medium transition-colors",
                  active
                    ? "text-white"
                    : "text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100"
                )}
              >
                {active && (
                  <motion.div
                    layoutId="nav-pill"
                    className="absolute inset-0 bg-accent rounded-full shadow-glow"
                    transition={{ type: "spring", stiffness: 380, damping: 30 }}
                  />
                )}
                <Icon className="w-4 h-4 relative z-10" />
                <span className="hidden md:inline relative z-10">{label}</span>
              </NavLink>
            );
          })}
        </div>
      </motion.nav>

      {/* Theme toggle — top right */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        className="fixed top-6 right-6 z-50"
      >
        <ThemeToggle />
      </motion.div>
    </>
  );
}