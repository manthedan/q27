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
#endif

// Spawn `powermetrics --samplers cpu_power,gpu_power,thermal -i 1000` writing
// to $Q27_BENCH_POWER_LOG. Returns true if a sampler is running. No-op
// (returns false) when the env is unset or this is not macOS.
inline bool bench_thermal_start() {
#if defined(__APPLE__)
    const char* log = std::getenv("Q27_BENCH_POWER_LOG");
    if (!log || !*log) return false;
    std::string out = std::string(log);
    // powermetrics needs root for some samplers on some macOS builds; run
    // best-effort and let the sampler's own stderr land in the log.
    std::string redir = out;
    std::string cmd = "exec powermetrics --samplers cpu_power,gpu_power,thermal "
                      "-i 1000 >> '" + redir + "' 2>&1";
    char* argv[] = {(char*)"/bin/sh", (char*)"-c", (char*)cmd.c_str(), nullptr};
    pid_t pid;
    if (posix_spawn(&pid, "/bin/sh", nullptr, nullptr, argv, environ) != 0) {
        std::fprintf(stderr, "[bench_env] powermetrics spawn failed (continuing)\n");
        return false;
    }
    g_power_pid = pid;
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
        // powermetrics is a long-running sampler; stop it at bench teardown.
        kill(g_power_pid, SIGTERM);
        g_power_pid = -1;
    }
#endif
}

// The line appended to the bench's stdout artifact. sampled = whether a
// powermetrics log is active this run.
inline void bench_power_line(bool sampled) {
#if defined(__APPLE__)
    const char* log = std::getenv("Q27_BENCH_POWER_LOG");
    if (sampled && log && *log)
        std::printf("power-state: powermetrics sampled, log %s\n", log);
    else
        std::printf("power-state: NOT SAMPLED (set Q27_BENCH_POWER_LOG to record)\n");
#else
    std::printf("power-state: n/a (non-macOS bench host)\n");
#endif
}

} // namespace q27bench
