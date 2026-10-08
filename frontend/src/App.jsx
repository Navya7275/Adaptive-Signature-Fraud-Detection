import { Routes, Route } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import FloatingNav from "./components/FloatingNav";

import Dashboard from "./pages/Dashboard";
import Enroll from "./pages/Enroll";
import Verify from "./pages/Verify";
import UserDetail from "./pages/UserDetail";
import Alerts from "./pages/Alerts";

export default function App() {
  return (
    <div className="min-h-screen">
      <FloatingNav />
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/enroll" element={<Enroll />} />
          <Route path="/verify" element={<Verify />} />
          <Route path="/users/:userId" element={<UserDetail />} />
          <Route path="/alerts" element={<Alerts />} />
        </Routes>
      </AnimatePresence>
    </div>
  );
}