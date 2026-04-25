"""Project-wide constants and semantic counter metadata."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
PROCESSED_RESULTS_DIR = RESULTS_DIR / "processed"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = RESULTS_DIR / "reports"
LOGS_DIR = RESULTS_DIR / "logs"

DEFAULT_INTERVAL_MS = 10
DEFAULT_SYNTHETIC_DURATION_MS = 1500
DEFAULT_SYNTHETIC_PHASE_MS = 150
DEFAULT_REPETITIONS = 3
DEFAULT_THREADS = [1, 2, 4, 8]
DEFAULT_CORRELATION_THRESHOLDS = [0.70, 0.80, 0.90]
DEFAULT_WINSOR_LIMITS = (0.01, 0.99)

METADATA_COLUMNS = [
    "timestamp_ms",
    "interval_duration_ms",
    "workload",
    "suite",
    "run_id",
    "threads",
    "cpu_or_core_id",
    "phase_label",
]

COUNTER_FAMILIES = {
    "instructions_retired": {
        "label": "Instructions retired",
        "portability": 10,
        "interpretability": 10,
        "collection_cost": 1,
        "preferred": {
            "intel": ["INST_RETIRED.ANY"],
            "amd": ["RETIRED_INST"],
            "arm": ["INST_RETIRED"],
            "generic": ["instructions"],
        },
    },
    "cycles": {
        "label": "Cycles",
        "portability": 10,
        "interpretability": 10,
        "collection_cost": 1,
        "preferred": {
            "intel": ["CPU_CLK_UNHALTED.THREAD"],
            "amd": ["CYCLES_NOT_IN_HALT"],
            "arm": ["CPU_CYCLES"],
            "generic": ["cycles", "cpu-cycles"],
        },
    },
    "branch_instructions": {
        "label": "Branch instructions",
        "portability": 9,
        "interpretability": 8,
        "collection_cost": 1,
        "preferred": {
            "intel": ["BR_INST_RETIRED.ALL_BRANCHES"],
            "amd": ["RETIRED_BR_INST"],
            "arm": ["BR_RETIRED"],
            "generic": ["branches", "branch-instructions"],
        },
    },
    "branch_mispredictions": {
        "label": "Branch mispredictions",
        "portability": 9,
        "interpretability": 9,
        "collection_cost": 1,
        "preferred": {
            "intel": ["BR_MISP_RETIRED.ALL_BRANCHES"],
            "amd": ["RETIRED_BR_INST_MISP"],
            "arm": ["BR_MIS_PRED_RETIRED"],
            "generic": ["branch-misses"],
        },
    },
    "l1d_loads": {
        "label": "L1 data loads",
        "portability": 7,
        "interpretability": 7,
        "collection_cost": 2,
        "preferred": {
            "intel": ["MEM_UOPS_RETIRED.ALL_LOADS"],
            "amd": ["LS_DISPATCH.LOADS"],
            "arm": ["L1D_CACHE"],
            "generic": ["L1-dcache-loads", "cpu/L1-dcache-loads/"],
        },
    },
    "l1d_stores": {
        "label": "L1 data stores",
        "portability": 6,
        "interpretability": 7,
        "collection_cost": 2,
        "preferred": {
            "intel": ["MEM_UOPS_RETIRED.ALL_STORES"],
            "amd": ["LS_DISPATCH.STORES"],
            "arm": ["L1D_CACHE_WB"],
            "generic": ["L1-dcache-stores", "cpu/L1-dcache-stores/"],
        },
    },
    "l2_misses": {
        "label": "L2 misses",
        "portability": 7,
        "interpretability": 8,
        "collection_cost": 2,
        "preferred": {
            "intel": ["L2_RQSTS.MISS"],
            "amd": ["L2 Miss"],
            "arm": ["L2D_CACHE_REFILL"],
            "generic": ["l2_rqsts.miss", "L2-dcache-load-misses", "cpu/L2-dcache-load-misses/"],
        },
    },
    "llc_references": {
        "label": "LLC references",
        "portability": 8,
        "interpretability": 8,
        "collection_cost": 2,
        "preferred": {
            "intel": ["LONGEST_LAT_CACHE.REFERENCE", "LLC_REFERENCES"],
            "amd": ["L3 Access"],
            "arm": ["L3D_CACHE"],
            "generic": ["cache-references", "LLC-loads", "cpu/LLC-loads/"],
        },
    },
    "llc_misses": {
        "label": "LLC misses",
        "portability": 8,
        "interpretability": 8,
        "collection_cost": 2,
        "preferred": {
            "intel": ["LONGEST_LAT_CACHE.MISS", "LLC_MISSES"],
            "amd": ["L3 Miss"],
            "arm": ["L3D_CACHE_REFILL"],
            "generic": ["cache-misses", "LLC-load-misses", "cpu/LLC-load-misses/"],
        },
    },
    "offcore_demand_data_reads": {
        "label": "Off-core / DRAM demand reads",
        "portability": 4,
        "interpretability": 7,
        "collection_cost": 3,
        "preferred": {
            "intel": ["OFFCORE_REQUESTS.DEMAND_DATA_RD"],
            "amd": ["DC Fills from Local Memory", "DC Fills from Remote Memory"],
            "arm": ["PMU_AXI_RD_REQ_EVENT"],
            "generic": [],
        },
    },
    "fp_arithmetic": {
        "label": "Floating-point arithmetic",
        "portability": 5,
        "interpretability": 8,
        "collection_cost": 2,
        "preferred": {
            "intel": ["FP_ARITH_INST_RETIRED.SCALAR_DOUBLE"],
            "amd": ["RETIRED_SSE_AVX_FLOPS"],
            "arm": ["FP_RETIRED"],
            "generic": [],
        },
    },
    "resource_stalls": {
        "label": "Resource stalls",
        "portability": 6,
        "interpretability": 6,
        "collection_cost": 2,
        "preferred": {
            "intel": ["RESOURCE_STALLS.ANY", "RESOURCE_STALLS"],
            "amd": ["DISPATCH_STALLS"],
            "arm": ["STALL_FRONTEND", "STALL_BACKEND"],
            "generic": ["stalled-cycles-frontend", "stalled-cycles-backend"],
        },
    },
    "memory_read_bandwidth": {
        "label": "Memory read bandwidth",
        "portability": 3,
        "interpretability": 9,
        "collection_cost": 4,
        "preferred": {
            "intel": ["READ"],
            "amd": ["Local DRAM Read Data Bytes"],
            "arm": ["PMU_AXI_RD_REQ_EVENT"],
            "generic": [],
        },
    },
    "memory_write_bandwidth": {
        "label": "Memory write bandwidth",
        "portability": 3,
        "interpretability": 9,
        "collection_cost": 4,
        "preferred": {
            "intel": ["WRITE"],
            "amd": ["Local DRAM Write Data Bytes"],
            "arm": ["PMU_AXI_WR_REQ_EVENT"],
            "generic": [],
        },
    },
    "total_memory_bandwidth": {
        "label": "Total memory bandwidth",
        "portability": 4,
        "interpretability": 9,
        "collection_cost": 4,
        "preferred": {
            "intel": ["READ", "WRITE"],
            "amd": ["Total Memory Bw"],
            "arm": ["PMU_AXI_RD_REQ_EVENT", "PMU_AXI_WR_REQ_EVENT"],
            "generic": [],
        },
    },
}

SYNTHETIC_WORKLOADS = [
    "compute",
    "memory",
    "cache",
    "branch",
    "fp",
    "mixed",
]
