-- Adaptive Aging-Aware Signature Fraud Detection System

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    enrollment_date TEXT NOT NULL,
    age_at_enrollment INTEGER,
    status TEXT DEFAULT 'active',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS signatures (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    file_hash TEXT,
    embedding BLOB,
    capture_date TEXT NOT NULL,
    is_reference INTEGER DEFAULT 0,
    similarity_score REAL,
    pressure_mean REAL,
    stroke_speed REAL,
    tremor_index REAL,
    stroke_consistency REAL,
    pen_lift_count INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS drift_profiles (
    id TEXT PRIMARY KEY,
    user_id TEXT UNIQUE NOT NULL,
    drift_rate REAL DEFAULT 0.0,
    drift_direction TEXT DEFAULT 'stable',
    volatility REAL DEFAULT 0.0,
    trend_consistency REAL DEFAULT 1.0,
    last_updated TEXT,
    baseline_embedding BLOB,
    expected_score REAL DEFAULT 1.0,
    tenure_months INTEGER DEFAULT 0,
    trust_buffer REAL DEFAULT 0.0,
    base_threshold REAL DEFAULT 0.80,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS verification_logs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    signature_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    raw_score REAL NOT NULL,
    adjusted_score REAL NOT NULL,
    threshold_used REAL NOT NULL,
    decision TEXT NOT NULL,
    drift_classification TEXT,
    confidence REAL,
    reason TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (signature_id) REFERENCES signatures(id)
);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    resolved INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sig_user ON signatures(user_id);
CREATE INDEX IF NOT EXISTS idx_sig_date ON signatures(capture_date);
CREATE INDEX IF NOT EXISTS idx_vlog_user ON verification_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_alert_user ON alerts(user_id);