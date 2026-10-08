import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 30000,
});

// ── Users ──
export const listUsers = () => api.get("/api/users/").then((r) => r.data);

export const getUser = (id) => api.get(`/api/users/${id}`).then((r) => r.data);

export const getUserHistory = (id) =>
  api.get(`/api/users/${id}/history`).then((r) => r.data);

export const enrollUser = (name, age, files) => {
  const form = new FormData();
  form.append("name", name);
  form.append("age", age);
  files.forEach((f) => form.append("signatures", f));
  return api
    .post("/api/users/enroll", form, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    .then((r) => r.data);
};

// ── Verification ──
export const verifySignature = (userId, file) => {
  const form = new FormData();
  form.append("user_id", userId);
  form.append("signature", file);
  return api
    .post("/api/verify/", form, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    .then((r) => r.data);
};

export const getVerificationLogs = (userId, limit = 20) =>
  api.get(`/api/verify/logs/${userId}?limit=${limit}`).then((r) => r.data);

// ── Alerts ──
export const getAlerts = (resolved = false) =>
  api.get(`/api/verify/alerts?resolved=${resolved}`).then((r) => r.data);

export const resolveAlert = (alertId) =>
  api.put(`/api/verify/alerts/${alertId}/resolve`).then((r) => r.data);

export default api;