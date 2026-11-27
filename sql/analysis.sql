-- Per run, count all questions and publish accuracy once all manual grades exist.
SELECT r.id AS run_id, r.notes AS run_name, COUNT(q.id) AS questions,
       COUNT(q.correct) AS graded, COUNT(*) FILTER (WHERE q.correct) AS correct,
       CASE WHEN COUNT(q.correct) = COUNT(q.id) THEN
           ROUND(100.0 * COUNT(*) FILTER (WHERE q.correct) / NULLIF(COUNT(q.id), 0), 2)
       END AS accuracy_pct
FROM eval_runs r LEFT JOIN eval_results q ON q.run_id = r.id
WHERE CAST(:run_id AS integer) IS NULL OR r.id = :run_id
GROUP BY r.id ORDER BY r.id;

-- Group manual accuracy and grading coverage by category.
SELECT category, COUNT(*) AS questions, COUNT(correct) AS graded,
       CASE WHEN COUNT(correct) = COUNT(*) THEN
           ROUND(100.0 * COUNT(*) FILTER (WHERE correct) / COUNT(*), 2)
       END AS accuracy_pct
FROM eval_results WHERE CAST(:run_id AS integer) IS NULL OR run_id = :run_id
GROUP BY category ORDER BY category;

-- Count labeled errors and their share of all labeled errors.
SELECT failure_mode, COUNT(*) AS errors,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct
FROM eval_results WHERE correct IS FALSE AND failure_mode IS NOT NULL
AND (CAST(:run_id AS integer) IS NULL OR run_id = :run_id)
GROUP BY failure_mode ORDER BY errors DESC, failure_mode;

-- Count evidence hits on answerable questions and show missing measurements.
SELECT run_id, COUNT(*) FILTER (WHERE retrieval_hit) AS hits, COUNT(*) AS answerable,
       COUNT(retrieval_hit) AS measured,
       CASE WHEN COUNT(retrieval_hit) = COUNT(*) THEN
           ROUND(100.0 * COUNT(*) FILTER (WHERE retrieval_hit) / COUNT(*), 2)
       END AS hit_pct
FROM eval_results WHERE answerable AND (CAST(:run_id AS integer) IS NULL OR run_id = :run_id)
GROUP BY run_id ORDER BY run_id;

-- Interpolate latency percentiles in milliseconds for each run and split.
SELECT r.id AS run_id, r.split,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY q.latency_ms) AS p50_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY q.latency_ms) AS p95_ms
FROM eval_runs r JOIN eval_results q ON q.run_id = r.id
WHERE CAST(:run_id AS integer) IS NULL OR r.id = :run_id
GROUP BY r.id, r.split ORDER BY r.id;

-- Compare explicit baseline and later runs on the same dataset, split and question.
SELECT a.question_id, a.answer AS before_answer, b.answer AS after_answer
FROM eval_results a JOIN eval_results b ON b.question_id = a.question_id
JOIN eval_runs old_run ON old_run.id = a.run_id
JOIN eval_runs new_run ON new_run.id = b.run_id AND new_run.dataset_name = old_run.dataset_name
    AND new_run.split = old_run.split
WHERE a.run_id = :baseline_id AND b.run_id = :run_id AND a.correct IS FALSE AND b.correct IS TRUE
ORDER BY a.question_id;

-- Count messages per uploaded report and average only assistant response latency.
SELECT r.id AS report_id, r.filename, COUNT(m.id) AS messages,
       AVG(m.latency_ms) FILTER (WHERE m.role = 'assistant') AS avg_response_ms
FROM reports r LEFT JOIN chat_sessions s ON s.report_id = r.id
LEFT JOIN chat_messages m ON m.session_id = s.id GROUP BY r.id, r.filename ORDER BY r.id;
