// Shared bench-environment instrumentation (expert review 2 §4 thermal
// governance, adopted 2026-07-18 into QA_BEFORE_RELEASES.md section 6).
// Twice-burned by unrecorded machine state: the gate-3 16K wall anomaly
// (App Nap / power state on a bare-nohup overnight run) and the 12.66→10.57
// tok/s decode-ceiling drift mid-session (thermal). Every bench artifact
// that gates a ship decision must carry a thermal/power line so a
// contaminated wall is attributable after the fact.
//
// Two pieces:
//   bench_thermal_start() — spawn a best-effort powermetrics sampler writing
//     thermal_pressure + CPU/GPU/ANE power to a log file. OPT-IN via
//     Q27_BENCH_POWER_LOG=<path> (the QA harness sets it per leg); absent the
//     env it is a no-op so benches stay single-process and portable. Fails
//     open (powermetrics missing/unreadable -> stderr note, never a bench
//     failure). macOS-only; other platforms no-op.
//   bench_power_line() — a one-line summary appended to the bench's own
//     stdout artifact: powermetrics log path if sampling, else the explicit
//     "not sampled" marker the QA checklist greps for. The point is that a
//     wall number WITHOUT a power-state line is visibly incomplete.
//
// Per-token-bytes is reported separately by each bench (the decode bench
// already computes token_weight_bytes; the line is named so review-2's
// "counted bytes, not byte/unit fits" reconciliation reads it directly).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(__APPLE__)
#include <signal.h>
#include <spawn.h>
#include <sys/types.h>
#include <unistd.h>
extern char** environ;
#endif

namespace q27bench {

#if defined(__APPLE__)
inline pid_t g_power_pid = -1;
inline std::string g_power_log;
inline std::string g_therm_start, g_therm_end;
#endif

// One-line flattening of a pmset thermal dump for the artifact line.
inline std::string bench_therm_oneline(const std::string& dump) {
    std::string r;
    for (char c : dump) {
        if (c == '\n') { if (!r.empty() && r.back() != ' ') r += ' '; continue; }
        if (c == '\r') continue;
        r += c;
    }
    // collapse the pmset "Note: " prefixes
    std::string out;
    const std::string p = "Note: ";
    size_t i = 0;
    for (; (i = r.find(p)) != std::string::npos;) r.erase(i, p.size());
    return r;
}

// Spawn `powermetrics --samplers cpu_power,gpu_power,thermal -i 1000` writing
// to $Q27_BENCH_POWER_LOG. Returns true if a sampler is running. No-op
// (returns false) when the env is unset or this is not macOS.
// Capture the current thermal state (no root needed) into `out` by running
// `pmset -g therm`. Returns true if pmset answered. The QA harness calls this
// at bench start and end and diffs the two: a mid-bench thermal-pressure or
// CPU-power-status transition is exactly the contamination class that burned
// the gate-3 wall and the 12.66->10.57 ceiling drift.
inline bool bench_pmset_therm(std::string& out) {
#if defined(__APPLE__)
    FILE* p = popen("pmset -g therm 2>/dev/null", "r");
    if (!p) return false;
    char buf[256];
    out.clear();
    while (fgets(buf, sizeof buf, p)) out += buf;
    return pclose(p) != -1 && !out.empty();
#else
    (void)out; return false;
#endif
}

inline bool bench_thermal_start() {
#if defined(__APPLE__)
    const char* log = std::getenv("Q27_BENCH_POWER_LOG");
    if (!log || !*log) return false;
    // Always anchor the run's STARTING thermal state (root-free). This is
    // the record that survives when powermetrics cannot run.
    g_therm_start.clear();
    bench_pmset_therm(g_therm_start);
    // powermetrics needs root on this host; try it best-effort and treat a
    // superuser refusal as "sampler unavailable", not a bench failure.
    // Truncate the log first so bench_power_log_has_samples() judges THIS
    // leg, not a previous run's leftover (the >> redirect appends).
    std::string out = std::string(log);
    { FILE* t = std::fopen(out.c_str(), "w"); if (t) std::fclose(t); }
    std::string cmd = "exec powermetrics --samplers cpu_power,gpu_power,thermal "
                      "-i 1000 >> '" + out + "' 2>&1";
    char* argv[] = {(char*)"/bin/sh", (char*)"-c", (char*)cmd.c_str(), nullptr};
    pid_t pid;
    if (posix_spawn(&pid, "/bin/sh", nullptr, nullptr, argv, environ) != 0) {
        std::fprintf(stderr, "[bench_env] powermetrics spawn failed (pmset thermal anchor only)\n");
        return false;
    }
    g_power_pid = pid;
    g_power_log = out;
    std::fprintf(stderr, "[bench_env] powermetrics sampling -> %s (pid %d)\n",
                 log, (int)pid);
    return true;
#else
    return false;
#endif
}

inline void bench_thermal_stop() {
#if defined(__APPLE__)
    if (g_power_pid > 0) {
        kill(g_power_pid, SIGTERM);
        g_power_pid = -1;
    }
    g_therm_end.clear();
    bench_pmset_therm(g_therm_end);
#endif
}

// True if the powermetrics log captured real samples (more than the
// superuser-refusal line). Used to downgrade the report honestly.
inline bool bench_power_log_has_samples() {
#if defined(__APPLE__)
    if (g_power_log.empty()) return false;
    FILE* f = std::fopen(g_power_log.c_str(), "r");
    if (!f) return false;
    char buf[256];
    bool real = false;
    while (fgets(buf, sizeof buf, f)) {
        std::string line(buf);
        if (line.find("must be invoked as the superuser") != std::string::npos) continue;
        if (line.find_first_not_of(" \t\r\n") != std::string::npos) { real = true; break; }
    }
    std::fclose(f);
    return real;
#else
    return false;
#endif
}

// The line appended to the bench's stdout artifact. sampled = whether a
// powermetrics log is active this run. Reports three honest states:
// full powermetrics sampling, pmset-only anchor (powermetrics refused),
// or not sampled.
inline void bench_power_line(bool sampled) {
#if defined(__APPLE__)
    const char* log = std::getenv("Q27_BENCH_POWER_LOG");
    if (sampled && bench_power_log_has_samples())
        std::printf("power-state: powermetrics sampled, log %s\n", log);
    else if (sampled)
        std::printf("power-state: pmset thermal anchor only (powermetrics needs root), log %s\n",
                    log ? log : "");
    else
        std::printf("power-state: NOT SAMPLED (set Q27_BENCH_POWER_LOG to record)\n");
    // The start/end thermal anchor rides the artifact regardless: a
    // mid-bench thermal-pressure transition is the contamination class the
    // QA checklist guards against, and pmset needs no root.
    if (!g_therm_start.empty() || !g_therm_end.empty())
        std::printf("thermal: start[%s] end[%s]\n",
                    bench_therm_oneline(g_therm_start).c_str(),
                    bench_therm_oneline(g_therm_end).c_str());
#else
    std::printf("power-state: n/a (non-macOS bench host)\n");
#endif
}

} // namespace q27bench
