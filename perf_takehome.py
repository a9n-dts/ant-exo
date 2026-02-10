"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import os
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)
    
    def init_instr_bundle(self):

        instr_slot = {
            "alu": [],
            "valu": [],
            "load": [],
            "store": [],
            "flow": [],
            "debug": [],
        }

        instr_cnt = {
            "alu": 0,
            "valu": 0,
            "load": 0,
            "store": 0,
            "flow": 0,
            "debug": 0,
        }

        return instr_slot, instr_cnt


    def get_slot_deps(self, engine, slot):
        """Return (reads, writes) sets of scratch addresses for dependency tracking."""
        reads = set()
        writes = set()
        match engine:
            case "alu":
                op, dest, a1, a2 = slot
                reads.update([a1, a2])
                writes.add(dest)
            case "valu":
                match slot:
                    case ("vbroadcast", dest, src):
                        reads.add(src)
                        writes.update(range(dest, dest + VLEN))
                    case ("multiply_add", dest, a, b, c):
                        reads.update(range(a, a + VLEN))
                        reads.update(range(b, b + VLEN))
                        reads.update(range(c, c + VLEN))
                        writes.update(range(dest, dest + VLEN))
                    case (op, dest, a1, a2):
                        reads.update(range(a1, a1 + VLEN))
                        reads.update(range(a2, a2 + VLEN))
                        writes.update(range(dest, dest + VLEN))
            case "load":
                match slot:
                    case ("load", dest, addr):
                        reads.add(addr)
                        writes.add(dest)
                    case ("const", dest, val):
                        writes.add(dest)
                    case ("vload", dest, addr):
                        reads.add(addr)
                        writes.update(range(dest, dest + VLEN))
                    case ("load_offset", dest, addr, offset):
                        reads.add(addr + offset)
                        writes.add(dest + offset)
            case "store":
                match slot:
                    case ("store", addr, src):
                        reads.update([addr, src])
                    case ("vstore", addr, src):
                        reads.add(addr)
                        reads.update(range(src, src + VLEN))
            case "flow":
                match slot:
                    case ("select", dest, cond, a, b):
                        reads.update([cond, a, b])
                        writes.add(dest)
                    case ("add_imm", dest, a, imm):
                        reads.add(a)
                        writes.add(dest)
                    case ("vselect", dest, cond, a, b):
                        reads.update(range(cond, cond + VLEN))
                        reads.update(range(a, a + VLEN))
                        reads.update(range(b, b + VLEN))
                        writes.update(range(dest, dest + VLEN))
                    case ("coreid", dest):
                        writes.add(dest)
                    case ("cond_jump", cond, _):
                        reads.add(cond)
                    case ("cond_jump_rel", cond, _):
                        reads.add(cond)
                    case ("jump_indirect", addr):
                        reads.add(addr)
                    case ("trace_write", val):
                        reads.add(val)
            case "debug":
                match slot:
                    case ("compare", loc, key):
                        reads.add(loc)
                    case ("vcompare", loc, keys):
                        reads.update(range(loc, loc + VLEN))
        return reads, writes

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        instrs = []
        instr_slot, instr_cnt = self.init_instr_bundle()
        written_in_bundle = set()

        for engine, slot in slots:
            reads, writes = self.get_slot_deps(engine, slot)

            # RAW: new slot reads something written in current bundle
            has_raw = bool(reads & written_in_bundle)
            # WAW: new slot writes something already written in current bundle
            has_waw = bool(writes & written_in_bundle)
            # Engine slot limit exceeded
            would_exceed = instr_cnt[engine] >= SLOT_LIMITS[engine]

            if has_raw or has_waw or would_exceed:
                instrs.append(instr_slot)
                instr_slot, instr_cnt = self.init_instr_bundle()
                written_in_bundle = set()

            instr_slot[engine].append(slot)
            instr_cnt[engine] += 1
            written_in_bundle.update(writes)

        instrs.append(instr_slot)
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized kernel using vload/vstore/valu/vselect.
        Processes VLEN=8 batch elements per vector instruction.
        """
        assert batch_size % VLEN == 0, f"batch_size must be a multiple of VLEN={VLEN}"
        n_groups = batch_size // VLEN
        max_pipeline_groups = int(os.environ.get("PK_PIPE_GROUPS", "16"))
        max_pipeline_groups = max(8, min(max_pipeline_groups, n_groups))

        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        tmp4 = self.alloc_scratch("tmp4")
        tmp5 = self.alloc_scratch("tmp5")
        tmp6 = self.alloc_scratch("tmp6")

        # Scalar constants and derived layout pointers
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        forest_values_p_const = 7
        inp_values_p_const = forest_values_p_const + n_nodes + batch_size
        batch_size_const = self.scratch_const(batch_size)
        forest_values_p = self.scratch_const(forest_values_p_const)
        inp_values_p = self.scratch_const(inp_values_p_const)

        tmp7 = self.alloc_scratch("tmp7")
        tmp8 = self.alloc_scratch("tmp8")
        tmp9 = self.alloc_scratch("tmp9")
        tmp10 = self.alloc_scratch("tmp10")
        tmp11 = self.alloc_scratch("tmp11")
        tmp12 = self.alloc_scratch("tmp12")
        tmp13 = self.alloc_scratch("tmp13")
        tmp14 = self.alloc_scratch("tmp14")

        # Vector work registers for six interleaved groups (VLEN=8 words each)
        v_idx_a = self.alloc_scratch("v_idx_a", VLEN)
        v_val_a = self.alloc_scratch("v_val_a", VLEN)
        v_node_val_a = self.alloc_scratch("v_node_val_a", VLEN)
        v_tmp1_a = self.alloc_scratch("v_tmp1_a", VLEN)
        v_tmp2_a = self.alloc_scratch("v_tmp2_a", VLEN)
        v_addr_a = self.alloc_scratch("v_addr_a", VLEN)

        v_idx_b = self.alloc_scratch("v_idx_b", VLEN)
        v_val_b = self.alloc_scratch("v_val_b", VLEN)
        v_node_val_b = self.alloc_scratch("v_node_val_b", VLEN)
        v_tmp1_b = self.alloc_scratch("v_tmp1_b", VLEN)
        v_tmp2_b = self.alloc_scratch("v_tmp2_b", VLEN)
        v_addr_b = self.alloc_scratch("v_addr_b", VLEN)

        v_idx_c = self.alloc_scratch("v_idx_c", VLEN)
        v_val_c = self.alloc_scratch("v_val_c", VLEN)
        v_node_val_c = self.alloc_scratch("v_node_val_c", VLEN)
        v_tmp1_c = self.alloc_scratch("v_tmp1_c", VLEN)
        v_tmp2_c = self.alloc_scratch("v_tmp2_c", VLEN)
        v_addr_c = self.alloc_scratch("v_addr_c", VLEN)

        v_idx_d = self.alloc_scratch("v_idx_d", VLEN)
        v_val_d = self.alloc_scratch("v_val_d", VLEN)
        v_node_val_d = self.alloc_scratch("v_node_val_d", VLEN)
        v_tmp1_d = self.alloc_scratch("v_tmp1_d", VLEN)
        v_tmp2_d = self.alloc_scratch("v_tmp2_d", VLEN)
        v_addr_d = self.alloc_scratch("v_addr_d", VLEN)

        v_idx_e = self.alloc_scratch("v_idx_e", VLEN)
        v_val_e = self.alloc_scratch("v_val_e", VLEN)
        v_node_val_e = self.alloc_scratch("v_node_val_e", VLEN)
        v_tmp1_e = self.alloc_scratch("v_tmp1_e", VLEN)
        v_tmp2_e = self.alloc_scratch("v_tmp2_e", VLEN)
        v_addr_e = self.alloc_scratch("v_addr_e", VLEN)

        v_idx_f = self.alloc_scratch("v_idx_f", VLEN)
        v_val_f = self.alloc_scratch("v_val_f", VLEN)
        v_node_val_f = self.alloc_scratch("v_node_val_f", VLEN)
        v_tmp1_f = self.alloc_scratch("v_tmp1_f", VLEN)
        v_tmp2_f = self.alloc_scratch("v_tmp2_f", VLEN)
        v_addr_f = self.alloc_scratch("v_addr_f", VLEN)

        v_idx_g = self.alloc_scratch("v_idx_g", VLEN)
        v_val_g = self.alloc_scratch("v_val_g", VLEN)
        v_node_val_g = self.alloc_scratch("v_node_val_g", VLEN)
        v_tmp1_g = self.alloc_scratch("v_tmp1_g", VLEN)
        v_tmp2_g = self.alloc_scratch("v_tmp2_g", VLEN)
        v_addr_g = self.alloc_scratch("v_addr_g", VLEN)

        v_idx_h = self.alloc_scratch("v_idx_h", VLEN)
        v_val_h = self.alloc_scratch("v_val_h", VLEN)
        v_node_val_h = self.alloc_scratch("v_node_val_h", VLEN)
        v_tmp1_h = self.alloc_scratch("v_tmp1_h", VLEN)
        v_tmp2_h = self.alloc_scratch("v_tmp2_h", VLEN)
        v_addr_h = self.alloc_scratch("v_addr_h", VLEN)

        # Vector constants (broadcast from scalars)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_fvp = self.alloc_scratch("v_fvp", VLEN)
        v_root = self.alloc_scratch("v_root", VLEN)
        v_d1_l = self.alloc_scratch("v_d1_l", VLEN)
        v_d1_r = self.alloc_scratch("v_d1_r", VLEN)
        v_d1_delta = self.alloc_scratch("v_d1_delta", VLEN)
        root_node = self.alloc_scratch("root_node")
        node_addr = self.alloc_scratch("node_addr")
        node_d1_l = self.alloc_scratch("node_d1_l")
        node_d1_r = self.alloc_scratch("node_d1_r")

        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_fvp, forest_values_p))
        self.add("load", ("load", root_node, forest_values_p))
        self.add("valu", ("vbroadcast", v_root, root_node))
        self.add("flow", ("add_imm", node_addr, forest_values_p, 1))
        self.add("load", ("load", node_d1_l, node_addr))
        self.add("flow", ("add_imm", node_addr, forest_values_p, 2))
        self.add("load", ("load", node_d1_r, node_addr))
        self.add("valu", ("vbroadcast", v_d1_l, node_d1_l))
        self.add("valu", ("vbroadcast", v_d1_r, node_d1_r))
        self.add("valu", ("-", v_d1_delta, v_d1_l, v_d1_r))

        # Depth-2 pre-computation: load tree[3..6], compute deltas
        node_d2_3 = self.alloc_scratch("node_d2_3")
        node_d2_4 = self.alloc_scratch("node_d2_4")
        node_d2_5 = self.alloc_scratch("node_d2_5")
        node_d2_6 = self.alloc_scratch("node_d2_6")
        delta_lo_s = self.alloc_scratch("delta_lo_s")
        delta_hi_s = self.alloc_scratch("delta_hi_s")
        v_d2_tree3 = self.alloc_scratch("v_d2_tree3", VLEN)
        v_d2_tree5 = self.alloc_scratch("v_d2_tree5", VLEN)
        v_d2_delta_lo = self.alloc_scratch("v_d2_delta_lo", VLEN)
        v_d2_delta_hi = self.alloc_scratch("v_d2_delta_hi", VLEN)
        v_d2_offset = self.alloc_scratch("v_d2_offset", VLEN)
        three_const = self.scratch_const(3)

        self.add("flow", ("add_imm", node_addr, forest_values_p, 3))
        self.add("load", ("load", node_d2_3, node_addr))
        self.add("flow", ("add_imm", node_addr, forest_values_p, 4))
        self.add("load", ("load", node_d2_4, node_addr))
        self.add("flow", ("add_imm", node_addr, forest_values_p, 5))
        self.add("load", ("load", node_d2_5, node_addr))
        self.add("flow", ("add_imm", node_addr, forest_values_p, 6))
        self.add("load", ("load", node_d2_6, node_addr))
        self.add("alu", ("-", delta_lo_s, node_d2_4, node_d2_3))
        self.add("alu", ("-", delta_hi_s, node_d2_6, node_d2_5))
        self.add("valu", ("vbroadcast", v_d2_tree3, node_d2_3))
        self.add("valu", ("vbroadcast", v_d2_tree5, node_d2_5))
        self.add("valu", ("vbroadcast", v_d2_delta_lo, delta_lo_s))
        self.add("valu", ("vbroadcast", v_d2_delta_hi, delta_hi_s))
        self.add("valu", ("vbroadcast", v_d2_offset, three_const))

        # Broadcast hash stage constants to vectors and precompute simplified stages.
        # For stages of the form (a + c1) + (a << k), use multiply_add:
        # a = a * (2^k + 1) + c1
        hash_stage_plan = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                v_mul = self.alloc_scratch(f"v_h{hi}_mul", VLEN)
                v_add = self.alloc_scratch(f"v_h{hi}_add", VLEN)
                s_mul = self.scratch_const((1 << val3) + 1)
                s_add = self.scratch_const(val1)
                self.add("valu", ("vbroadcast", v_mul, s_mul))
                self.add("valu", ("vbroadcast", v_add, s_add))
                hash_stage_plan.append(("muladd", v_mul, v_add))
            else:
                v_c1 = self.alloc_scratch(f"v_h{hi}_c1", VLEN)
                v_c3 = self.alloc_scratch(f"v_h{hi}_c3", VLEN)
                s_c1 = self.scratch_const(val1)
                s_c3 = self.scratch_const(val3)
                self.add("valu", ("vbroadcast", v_c1, s_c1))
                self.add("valu", ("vbroadcast", v_c3, s_c3))
                hash_stage_plan.append(("normal", op1, op2, op3, v_c1, v_c3))

        # Loop variables
        loop_g_off = self.alloc_scratch("loop_g_off")
        cond = self.alloc_scratch("cond")
        vlen_const = self.scratch_const(VLEN)
        two_vlen = 2 * VLEN
        three_vlen = 3 * VLEN
        eight_vlen = 8 * VLEN

        val_ptrs = [tmp2, tmp4, tmp6, tmp8, tmp10, tmp12, tmp13, tmp14]
        for i in range(max_pipeline_groups - 8):
            val_ptrs.append(self.alloc_scratch(f"val_ptr_{i}"))

        groups_all = [
            {
                "idx": v_idx_a,
                "val": v_val_a,
                "node": v_node_val_a,
                "tmp1": v_tmp1_a,
                "tmp2": v_tmp2_a,
                "addr": v_addr_a,
            },
            {
                "idx": v_idx_b,
                "val": v_val_b,
                "node": v_node_val_b,
                "tmp1": v_tmp1_b,
                "tmp2": v_tmp2_b,
                "addr": v_addr_b,
            },
            {
                "idx": v_idx_c,
                "val": v_val_c,
                "node": v_node_val_c,
                "tmp1": v_tmp1_c,
                "tmp2": v_tmp2_c,
                "addr": v_addr_c,
            },
            {
                "idx": v_idx_d,
                "val": v_val_d,
                "node": v_node_val_d,
                "tmp1": v_tmp1_d,
                "tmp2": v_tmp2_d,
                "addr": v_addr_d,
            },
            {
                "idx": v_idx_e,
                "val": v_val_e,
                "node": v_node_val_e,
                "tmp1": v_tmp1_e,
                "tmp2": v_tmp2_e,
                "addr": v_addr_e,
            },
            {
                "idx": v_idx_f,
                "val": v_val_f,
                "node": v_node_val_f,
                "tmp1": v_tmp1_f,
                "tmp2": v_tmp2_f,
                "addr": v_addr_f,
            },
            {
                "idx": v_idx_g,
                "val": v_val_g,
                "node": v_node_val_g,
                "tmp1": v_tmp1_g,
                "tmp2": v_tmp2_g,
                "addr": v_addr_g,
            },
            {
                "idx": v_idx_h,
                "val": v_val_h,
                "node": v_node_val_h,
                "tmp1": v_tmp1_h,
                "tmp2": v_tmp2_h,
                "addr": v_addr_h,
            },
        ]
        for i in range(8, max_pipeline_groups):
            groups_all.append(
                {
                    "idx": self.alloc_scratch(f"v_idx_{i}", VLEN),
                    "val": self.alloc_scratch(f"v_val_{i}", VLEN),
                    "node": self.alloc_scratch(f"v_node_val_{i}", VLEN),
                    "tmp1": self.alloc_scratch(f"v_tmp1_{i}", VLEN),
                    "tmp2": self.alloc_scratch(f"v_tmp2_{i}", VLEN),
                    "addr": self.alloc_scratch(f"v_addr_{i}", VLEN),
                }
            )
        wrap_period = forest_height + 1
        main_group_width = max_pipeline_groups
        env_main_group_width = os.environ.get("PK_MAIN_GROUPS")
        if env_main_group_width is not None:
            main_group_width = int(env_main_group_width)
        main_group_width = max(2, min(main_group_width, len(groups_all)))
        if main_group_width % 2 != 0:
            main_group_width -= 1

        def chunked(cycles, ops):
            for i in range(0, len(ops), SLOT_LIMITS["valu"]):
                cycles.append(ops[i : i + SLOT_LIMITS["valu"]])

        def gather_mode(round_idx):
            depth = round_idx % wrap_period
            if depth == 0:
                return "root"
            if depth == 1:
                return "d1"
            if depth == 2:
                return "d2"
            return "gather"

        def gather_slots_inphase(groups, round_idx):
            body = []
            mode = gather_mode(round_idx)
            if mode == "root":
                for g in groups:
                    body.append(("valu", ("^", g["val"], g["val"], v_root)))
                return body
            if mode == "d1":
                for g in groups:
                    body.append(("valu", ("&", g["tmp1"], g["idx"], v_one)))
                for g in groups:
                    body.append(
                        ("valu", ("multiply_add", g["node"], g["tmp1"], v_d1_delta, v_d1_r))
                    )
                for g in groups:
                    body.append(("valu", ("^", g["val"], g["val"], g["node"])))
                return body
            if mode == "d2":
                for g in groups:
                    body.append(("valu", ("-", g["tmp1"], g["idx"], v_d2_offset)))
                for g in groups:
                    body.append(("valu", ("&", g["tmp2"], g["tmp1"], v_one)))
                for g in groups:
                    body.append(("valu", (">>", g["addr"], g["tmp1"], v_one)))
                for g in groups:
                    body.append(("valu", ("multiply_add", g["node"], g["tmp2"], v_d2_delta_lo, v_d2_tree3)))
                for g in groups:
                    body.append(("valu", ("multiply_add", g["tmp1"], g["tmp2"], v_d2_delta_hi, v_d2_tree5)))
                for g in groups:
                    body.append(("valu", ("-", g["tmp2"], g["tmp1"], g["node"])))
                for g in groups:
                    body.append(("valu", ("multiply_add", g["node"], g["addr"], g["tmp2"], g["node"])))
                for g in groups:
                    body.append(("valu", ("^", g["val"], g["val"], g["node"])))
                return body
            for g in groups:
                body.append(("valu", ("+", g["addr"], v_fvp, g["idx"])))
            for offset in range(VLEN):
                for g in groups:
                    body.append(("load", ("load_offset", g["node"], g["addr"], offset)))
            for g in groups:
                body.append(("valu", ("^", g["val"], g["val"], g["node"])))
            return body

        def append_round_ops_inphase(body, groups, round_idx):
            body.extend(gather_slots_inphase(groups, round_idx))

            # In-phase hash packing across all active groups.
            for stage in hash_stage_plan:
                if stage[0] == "muladd":
                    _, v_mul, v_add = stage
                    for g in groups:
                        body.append(
                            ("valu", ("multiply_add", g["val"], g["val"], v_mul, v_add))
                        )
                else:
                    _, op1, op2, op3, vc1, vc3 = stage
                    for g in groups:
                        body.append(("valu", (op1, g["tmp1"], g["val"], vc1)))
                        body.append(("valu", (op3, g["tmp2"], g["val"], vc3)))
                    for g in groups:
                        body.append(("valu", (op2, g["val"], g["tmp1"], g["tmp2"])))

            # Assumes all initial indices are 0, so wrap rounds are deterministic.
            if (round_idx + 1) % wrap_period == 0:
                for g in groups:
                    body.append(("valu", ("^", g["idx"], g["idx"], g["idx"])))
            else:
                for g in groups:
                    body.append(("valu", ("&", g["tmp1"], g["val"], v_one)))
                for g in groups:
                    body.append(
                        ("valu", ("multiply_add", g["idx"], g["idx"], v_two, v_one))
                    )
                for g in groups:
                    body.append(("valu", ("+", g["idx"], g["idx"], g["tmp1"])))

        def emit_bundle(body, alu=None, valu=None, load=None, store=None, flow=None):
            bundle = {}
            if alu:
                bundle["alu"] = alu
            if valu:
                bundle["valu"] = valu
            if load:
                bundle["load"] = load
            if store:
                bundle["store"] = store
            if flow:
                bundle["flow"] = flow
            if bundle:
                body.append(bundle)

        def emit_valu_chunked(body, ops):
            for i in range(0, len(ops), SLOT_LIMITS["valu"]):
                emit_bundle(body, valu=ops[i : i + SLOT_LIMITS["valu"]])

        def add_valu_node(nodes, op, deps=None, earliest=0):
            nodes.append(
                {
                    "op": op,
                    "deps": [] if deps is None else deps,
                    "earliest": earliest,
                    "order": len(nodes),
                }
            )
            return len(nodes) - 1

        def schedule_valu_nodes(nodes):
            if not nodes:
                return []

            children = [[] for _ in nodes]
            for nid, node in enumerate(nodes):
                for dep in node["deps"]:
                    children[dep].append(nid)

            crit = [1] * len(nodes)
            for nid in range(len(nodes) - 1, -1, -1):
                if children[nid]:
                    crit[nid] = 1 + max(crit[cid] for cid in children[nid])

            pending = set(range(len(nodes)))
            done_cycle = {}
            cycles = []
            while pending:
                cycle_idx = len(cycles)
                ready = []
                for nid in pending:
                    node = nodes[nid]
                    if node["earliest"] > cycle_idx:
                        continue
                    if all(
                        dep in done_cycle and done_cycle[dep] < cycle_idx
                        for dep in node["deps"]
                    ):
                        ready.append(nid)

                if not ready:
                    cycles.append([])
                    continue

                ready.sort(key=lambda nid: (-crit[nid], nodes[nid]["order"]))
                chosen = ready[: SLOT_LIMITS["valu"]]
                cycles.append([nodes[nid]["op"] for nid in chosen])
                for nid in chosen:
                    pending.remove(nid)
                    done_cycle[nid] = cycle_idx

            while cycles and not cycles[-1]:
                cycles.pop()
            return cycles

        def append_hash_nodes(nodes, groups, round_idx, need_next_addr):
            prev_val = [None] * len(groups)
            prev_idx = [None] * len(groups)

            for stage in hash_stage_plan:
                if stage[0] == "muladd":
                    _, v_mul, v_add = stage
                    for gi, g in enumerate(groups):
                        deps = [] if prev_val[gi] is None else [prev_val[gi]]
                        prev_val[gi] = add_valu_node(
                            nodes,
                            ("multiply_add", g["val"], g["val"], v_mul, v_add),
                            deps=deps,
                        )
                else:
                    _, op1, op2, op3, vc1, vc3 = stage
                    for gi, g in enumerate(groups):
                        deps = [] if prev_val[gi] is None else [prev_val[gi]]
                        n1 = add_valu_node(
                            nodes, (op1, g["tmp1"], g["val"], vc1), deps=deps
                        )
                        n2 = add_valu_node(
                            nodes, (op3, g["tmp2"], g["val"], vc3), deps=deps
                        )
                        prev_val[gi] = add_valu_node(
                            nodes,
                            (op2, g["val"], g["tmp1"], g["tmp2"]),
                            deps=[n1, n2],
                        )

            should_update_idx = (
                round_idx < rounds - 1 and (round_idx + 1) % wrap_period != 0
            )
            if should_update_idx:
                if round_idx % wrap_period == 0:
                    # Root rounds always start with idx=0, so next idx is 1 + (val & 1).
                    for gi, g in enumerate(groups):
                        deps = [] if prev_val[gi] is None else [prev_val[gi]]
                        n1 = add_valu_node(
                            nodes, ("&", g["tmp1"], g["val"], v_one), deps=deps
                        )
                        prev_idx[gi] = add_valu_node(
                            nodes, ("+", g["idx"], g["tmp1"], v_one), deps=[n1]
                        )
                else:
                    for gi, g in enumerate(groups):
                        val_deps = [] if prev_val[gi] is None else [prev_val[gi]]
                        idx_deps = [] if prev_idx[gi] is None else [prev_idx[gi]]
                        n1 = add_valu_node(
                            nodes, ("&", g["tmp1"], g["val"], v_one), deps=val_deps
                        )
                        n2 = add_valu_node(
                            nodes,
                            ("multiply_add", g["idx"], g["idx"], v_two, v_one),
                            deps=idx_deps,
                        )
                        prev_idx[gi] = add_valu_node(
                            nodes, ("+", g["idx"], g["idx"], g["tmp1"]), deps=[n1, n2]
                        )

            if need_next_addr:
                for gi, g in enumerate(groups):
                    deps = [] if prev_idx[gi] is None else [prev_idx[gi]]
                    add_valu_node(nodes, ("+", g["addr"], v_fvp, g["idx"]), deps=deps)

        def gather_load_cycles(groups, gather_round_idx):
            if gather_mode(gather_round_idx) != "gather":
                return []
            load_cycles = []
            load_ops = []
            for offset in range(VLEN):
                for g in groups:
                    load_ops.append(("load_offset", g["node"], g["addr"], offset))
            for i in range(0, len(load_ops), SLOT_LIMITS["load"]):
                load_cycles.append(load_ops[i : i + SLOT_LIMITS["load"]])
            return load_cycles

        def append_gather_nodes(nodes, groups, gather_round_idx, load_ready_cycle):
            mode = gather_mode(gather_round_idx)
            if mode == "root":
                for g in groups:
                    add_valu_node(nodes, ("^", g["val"], g["val"], v_root))
                return
            if mode == "d1":
                for g in groups:
                    n1 = add_valu_node(nodes, ("&", g["tmp1"], g["idx"], v_one))
                    n2 = add_valu_node(
                        nodes,
                        ("multiply_add", g["node"], g["tmp1"], v_d1_delta, v_d1_r),
                        deps=[n1],
                    )
                    add_valu_node(nodes, ("^", g["val"], g["val"], g["node"]), deps=[n2])
                return

            if mode == "d2":
                for g in groups:
                    n1 = add_valu_node(nodes, ("-", g["tmp1"], g["idx"], v_d2_offset))
                    n2 = add_valu_node(nodes, ("&", g["tmp2"], g["tmp1"], v_one), deps=[n1])
                    n3 = add_valu_node(nodes, (">>", g["addr"], g["tmp1"], v_one), deps=[n1])
                    n4 = add_valu_node(
                        nodes,
                        ("multiply_add", g["node"], g["tmp2"], v_d2_delta_lo, v_d2_tree3),
                        deps=[n2],
                    )
                    n5 = add_valu_node(
                        nodes,
                        ("multiply_add", g["tmp1"], g["tmp2"], v_d2_delta_hi, v_d2_tree5),
                        deps=[n2, n3],  # n3 reads tmp1 (local); n5 overwrites it (pair_hi)
                    )
                    n6 = add_valu_node(nodes, ("-", g["tmp2"], g["tmp1"], g["node"]), deps=[n4, n5])
                    n7 = add_valu_node(
                        nodes,
                        ("multiply_add", g["node"], g["addr"], g["tmp2"], g["node"]),
                        deps=[n3, n6],
                    )
                    add_valu_node(nodes, ("^", g["val"], g["val"], g["node"]), deps=[n7])
                return

            # XOR depends on all gathers, so earliest start is next cycle after load cycles.
            for g in groups:
                add_valu_node(
                    nodes,
                    ("^", g["val"], g["val"], g["node"]),
                    earliest=load_ready_cycle,
                )

        def emit_gather_only(body, groups, gather_round_idx):
            load_cycles = gather_load_cycles(groups, gather_round_idx)
            nodes = []
            append_gather_nodes(nodes, groups, gather_round_idx, len(load_cycles))
            valu_cycles = schedule_valu_nodes(nodes)
            phase_len = max(len(load_cycles), len(valu_cycles))
            for ci in range(phase_len):
                loads = load_cycles[ci] if ci < len(load_cycles) else None
                vals = valu_cycles[ci] if ci < len(valu_cycles) else None
                emit_bundle(body, valu=vals, load=loads)

        def emit_overlap_phase(
            body, gather_groups, hash_groups, round_idx, need_next_addr, gather_round_idx
        ):
            load_cycles = gather_load_cycles(gather_groups, gather_round_idx)
            nodes = []
            append_hash_nodes(nodes, hash_groups, round_idx, need_next_addr)
            append_gather_nodes(nodes, gather_groups, gather_round_idx, len(load_cycles))
            valu_cycles = schedule_valu_nodes(nodes)

            phase_len = max(len(load_cycles), len(valu_cycles))
            for ci in range(phase_len):
                loads = load_cycles[ci] if ci < len(load_cycles) else None
                vals = valu_cycles[ci] if ci < len(valu_cycles) else None
                emit_bundle(body, valu=vals, load=loads)

        def emit_hash_only(body, groups, round_idx, need_next_addr):
            nodes = []
            append_hash_nodes(nodes, groups, round_idx, need_next_addr)
            for ops in schedule_valu_nodes(nodes):
                emit_bundle(body, valu=ops)

        def add_sched_node(nodes, engine, op, deps=None, decomposable=None, earliest=0):
            if decomposable is None:
                decomposable = engine == "valu" and op[0] not in ("multiply_add", "vbroadcast")
            nodes.append(
                {
                    "engine": engine,
                    "op": op,
                    "deps": [] if deps is None else deps,
                    "order": len(nodes),
                    "decomposable": decomposable,
                    "earliest": earliest,
                }
            )
            return len(nodes) - 1

        def schedule_mixed_nodes(nodes):
            if not nodes:
                return []
            alu_frag = int(os.environ.get("PK_ALU_FRAG", "4"))
            if alu_frag <= 0:
                alu_frag = 4
            if VLEN % alu_frag != 0:
                alu_frag = 4
            alu_frag = min(alu_frag, SLOT_LIMITS["alu"])

            children = [[] for _ in nodes]
            for nid, node in enumerate(nodes):
                for dep in node["deps"]:
                    children[dep].append(nid)

            crit = [1] * len(nodes)
            for nid in range(len(nodes) - 1, -1, -1):
                if children[nid]:
                    crit[nid] = 1 + max(crit[cid] for cid in children[nid])

            pending = set(range(len(nodes)))
            done_cycle = {}
            bundles = []
            # Track remaining lane fragments for decomposable VALU ops.
            # Splitting into smaller chunks helps pack ALU's 12 slots.
            alu_lane_frags = {}
            for nid, node in enumerate(nodes):
                if node["engine"] == "valu" and node.get("decomposable"):
                    frags = []
                    for lane_lo in range(0, VLEN, alu_frag):
                        frags.append((lane_lo, lane_lo + alu_frag))
                    alu_lane_frags[nid] = frags

            while pending:
                cycle_idx = len(bundles)
                ready_valu = []
                ready_load = []
                for nid in pending:
                    node = nodes[nid]
                    if node.get("earliest", 0) > cycle_idx:
                        continue
                    if all(
                        dep in done_cycle and done_cycle[dep] < cycle_idx
                        for dep in node["deps"]
                    ):
                        if node["engine"] == "valu":
                            ready_valu.append(nid)
                        else:
                            ready_load.append(nid)

                ready_valu.sort(key=lambda nid: (-crit[nid], nodes[nid]["order"]))
                ready_load.sort(key=lambda nid: (-crit[nid], nodes[nid]["order"]))

                ready_valu_nondec = [
                    nid for nid in ready_valu if not nodes[nid].get("decomposable")
                ]
                ready_valu_dec = [
                    nid for nid in ready_valu if nodes[nid].get("decomposable")
                ]

                chosen_valu = ready_valu_nondec[: SLOT_LIMITS["valu"]]
                chosen_load = ready_load[: SLOT_LIMITS["load"]]
                chosen_alu_frags = []
                chosen_alu_nodes = set()

                alu_slots_left = SLOT_LIMITS["alu"]
                while alu_slots_left >= alu_frag:
                    candidates = [
                        nid
                        for nid in ready_valu_dec
                        if nid not in chosen_valu and len(alu_lane_frags.get(nid, [])) > 0
                    ]
                    if not candidates:
                        break
                    # Finish partially-offloaded ops first to unblock dependents.
                    candidates.sort(
                        key=lambda nid: (
                            0 if len(alu_lane_frags[nid]) == 1 else 1,
                            -crit[nid],
                            nodes[nid]["order"],
                        )
                    )
                    nid = candidates[0]
                    lane_lo, lane_hi = alu_lane_frags[nid].pop(0)
                    frag_width = lane_hi - lane_lo
                    if frag_width > alu_slots_left:
                        alu_lane_frags[nid].insert(0, (lane_lo, lane_hi))
                        break
                    alu_slots_left -= frag_width
                    chosen_alu_frags.append((nid, lane_lo, lane_hi))
                    chosen_alu_nodes.add(nid)

                valu_slots_left = SLOT_LIMITS["valu"] - len(chosen_valu)
                if valu_slots_left > 0:
                    for nid in ready_valu_dec:
                        if nid in chosen_alu_nodes:
                            continue
                        chosen_valu.append(nid)
                        if len(chosen_valu) >= SLOT_LIMITS["valu"]:
                            break

                if not chosen_valu and not chosen_load and not chosen_alu_frags:
                    raise RuntimeError("No schedulable ops in mixed scheduler")

                alu_ops = []
                for nid, lane_lo, lane_hi in chosen_alu_frags:
                    op, dest, a1, a2 = nodes[nid]["op"]
                    for lane in range(lane_lo, lane_hi):
                        alu_ops.append((op, dest + lane, a1 + lane, a2 + lane))

                bundles.append(
                    {
                        "valu": [nodes[nid]["op"] for nid in chosen_valu],
                        "load": [nodes[nid]["op"] for nid in chosen_load],
                        "alu": alu_ops,
                    }
                )
                completed_alu_nodes = [
                    nid
                    for nid in chosen_alu_nodes
                    if len(alu_lane_frags.get(nid, [])) == 0
                ]
                for nid in chosen_valu + chosen_load + completed_alu_nodes:
                    pending.remove(nid)
                    done_cycle[nid] = cycle_idx
            return bundles

        def add_mixed_hazard_deps(nodes):
            """
            Add RAW/WAW/WAR dependencies based on scratch addresses.
            This makes scheduling robust when we aggressively remap decomposable
            VALU ops to ALU without relying on hand-wired deps only.
            """
            last_writer = {}
            last_readers = defaultdict(set)

            for nid, node in enumerate(nodes):
                reads, writes = self.get_slot_deps(node["engine"], node["op"])
                deps = set(node["deps"])

                for addr in reads:
                    writer = last_writer.get(addr)
                    if writer is not None:
                        deps.add(writer)

                for addr in writes:
                    writer = last_writer.get(addr)
                    if writer is not None:
                        deps.add(writer)
                    deps.update(last_readers[addr])

                node["deps"] = list(deps)

                for addr in reads:
                    last_readers[addr].add(nid)

                for addr in writes:
                    last_writer[addr] = nid
                    last_readers[addr].clear()

        def build_octet_schedule(groups):
            nodes = []
            val_ready = [None] * len(groups)
            idx_ready = [None] * len(groups)
            skew_gap = int(os.environ.get("PK_SKEW_GAP", "0"))
            skew_period = max(1, int(os.environ.get("PK_SKEW_PERIOD", "4")))

            def deps_of(*deps):
                return [dep for dep in deps if dep is not None]

            for round_idx in range(rounds):
                mode = gather_mode(round_idx)
                for gi, g in enumerate(groups):
                    start_earliest = 0
                    if round_idx == 0 and skew_gap > 0:
                        start_earliest = (gi % skew_period) * skew_gap
                    if mode == "root":
                        val_ready[gi] = add_sched_node(
                            nodes,
                            "valu",
                            ("^", g["val"], g["val"], v_root),
                            deps=deps_of(val_ready[gi]),
                            earliest=start_earliest,
                        )
                    elif mode == "d1":
                        n1 = add_sched_node(
                            nodes,
                            "valu",
                            ("&", g["tmp1"], g["idx"], v_one),
                            deps=deps_of(idx_ready[gi]),
                            earliest=start_earliest,
                        )
                        n2 = add_sched_node(
                            nodes,
                            "valu",
                            ("multiply_add", g["node"], g["tmp1"], v_d1_delta, v_d1_r),
                            deps=[n1],
                        )
                        val_ready[gi] = add_sched_node(
                            nodes,
                            "valu",
                            ("^", g["val"], g["val"], g["node"]),
                            deps=deps_of(val_ready[gi], n2),
                        )
                    elif mode == "d2":
                        n1 = add_sched_node(
                            nodes,
                            "valu",
                            ("-", g["tmp1"], g["idx"], v_d2_offset),
                            deps=deps_of(idx_ready[gi]),
                            earliest=start_earliest,
                        )
                        n2 = add_sched_node(
                            nodes,
                            "valu",
                            ("&", g["tmp2"], g["tmp1"], v_one),
                            deps=[n1],
                        )
                        n3 = add_sched_node(
                            nodes,
                            "valu",
                            (">>", g["addr"], g["tmp1"], v_one),
                            deps=[n1],
                        )
                        n4 = add_sched_node(
                            nodes,
                            "valu",
                            ("multiply_add", g["node"], g["tmp2"], v_d2_delta_lo, v_d2_tree3),
                            deps=[n2],
                        )
                        n5 = add_sched_node(
                            nodes,
                            "valu",
                            ("multiply_add", g["tmp1"], g["tmp2"], v_d2_delta_hi, v_d2_tree5),
                            deps=[n2, n3],  # n3 reads tmp1 (local); n5 overwrites it (pair_hi)
                        )
                        n6 = add_sched_node(
                            nodes,
                            "valu",
                            ("-", g["tmp2"], g["tmp1"], g["node"]),
                            deps=[n4, n5],
                        )
                        n7 = add_sched_node(
                            nodes,
                            "valu",
                            ("multiply_add", g["node"], g["addr"], g["tmp2"], g["node"]),
                            deps=[n3, n6],
                        )
                        val_ready[gi] = add_sched_node(
                            nodes,
                            "valu",
                            ("^", g["val"], g["val"], g["node"]),
                            deps=deps_of(val_ready[gi], n7),
                        )
                    else:
                        n_addr = add_sched_node(
                            nodes,
                            "valu",
                            ("+", g["addr"], v_fvp, g["idx"]),
                            deps=deps_of(idx_ready[gi]),
                            earliest=start_earliest,
                        )
                        load_nodes = []
                        for offset in range(VLEN):
                            load_nodes.append(
                                add_sched_node(
                                    nodes,
                                    "load",
                                    ("load_offset", g["node"], g["addr"], offset),
                                    deps=[n_addr],
                                )
                            )
                        val_ready[gi] = add_sched_node(
                            nodes,
                            "valu",
                            ("^", g["val"], g["val"], g["node"]),
                            deps=deps_of(val_ready[gi], *load_nodes),
                        )

                    for stage in hash_stage_plan:
                        if stage[0] == "muladd":
                            _, v_mul, v_add = stage
                            val_ready[gi] = add_sched_node(
                                nodes,
                                "valu",
                                ("multiply_add", g["val"], g["val"], v_mul, v_add),
                                deps=deps_of(val_ready[gi]),
                            )
                        else:
                            _, op1, op2, op3, vc1, vc3 = stage
                            n1 = add_sched_node(
                                nodes,
                                "valu",
                                (op1, g["tmp1"], g["val"], vc1),
                                deps=deps_of(val_ready[gi]),
                            )
                            n2 = add_sched_node(
                                nodes,
                                "valu",
                                (op3, g["tmp2"], g["val"], vc3),
                                deps=deps_of(val_ready[gi]),
                            )
                            val_ready[gi] = add_sched_node(
                                nodes,
                                "valu",
                                (op2, g["val"], g["tmp1"], g["tmp2"]),
                                deps=[n1, n2],
                            )

                    should_update_idx = (
                        round_idx < rounds - 1 and (round_idx + 1) % wrap_period != 0
                    )
                    if should_update_idx:
                        if round_idx % wrap_period == 0:
                            # Root rounds always start with idx=0.
                            n1 = add_sched_node(
                                nodes,
                                "valu",
                                ("&", g["tmp1"], g["val"], v_one),
                                deps=deps_of(val_ready[gi]),
                            )
                            idx_ready[gi] = add_sched_node(
                                nodes,
                                "valu",
                                ("+", g["idx"], g["tmp1"], v_one),
                                deps=[n1],
                            )
                        else:
                            n1 = add_sched_node(
                                nodes,
                                "valu",
                                ("&", g["tmp1"], g["val"], v_one),
                                deps=deps_of(val_ready[gi]),
                            )
                            n2 = add_sched_node(
                                nodes,
                                "valu",
                                ("multiply_add", g["idx"], g["idx"], v_two, v_one),
                                deps=deps_of(idx_ready[gi]),
                            )
                            idx_ready[gi] = add_sched_node(
                                nodes,
                                "valu",
                                ("+", g["idx"], g["idx"], g["tmp1"]),
                                deps=[n1, n2],
                            )

            add_mixed_hazard_deps(nodes)
            return schedule_mixed_nodes(nodes)

        def emit_tail_groups(group_count, g_off_words):
            groups = groups_all[:group_count]
            g_off_const = self.scratch_const(g_off_words)
            tail_body = []

            tail_body.append(
                ("alu", ("+", val_ptrs[0], inp_values_p, g_off_const))
            )
            for gi in range(1, group_count):
                tail_body.append(
                    ("alu", ("+", val_ptrs[gi], val_ptrs[gi - 1], vlen_const))
                )

            # Assumes all initial indices are 0.
            tail_body.append(
                (
                    "valu",
                    [("^", g["idx"], g["idx"], g["idx"]) for g in groups][0],
                )
            )
            for g in groups[1:]:
                tail_body.append(("valu", ("^", g["idx"], g["idx"], g["idx"])))

            for gi, g in enumerate(groups):
                tail_body.append(("load", ("vload", g["val"], val_ptrs[gi])))

            for round_idx in range(rounds):
                append_round_ops_inphase(tail_body, groups, round_idx)

            for gi, g in enumerate(groups):
                tail_body.append(("store", ("vstore", val_ptrs[gi], g["val"])))

            self.instrs.extend(self.build(tail_body))

        # Octet loop scheduled from a single dependency graph across all
        # 8 groups so load-heavy gathers and hash chains can interleave freely.
        octet_groups = (n_groups // main_group_width) * main_group_width
        if octet_groups > 0:
            self.add("load", ("const", loop_g_off, 0))
            active_groups = groups_all[:main_group_width]
            octet_batch = octet_groups * VLEN
            loop_bound = (
                batch_size_const
                if octet_batch == batch_size
                else self.scratch_const(octet_batch)
            )

            octet_body = []

            emit_bundle(
                octet_body,
                alu=[
                    ("+", val_ptrs[0], inp_values_p, loop_g_off),
                ],
            )
            for gi in range(1, main_group_width):
                emit_bundle(
                    octet_body,
                    alu=[
                        ("+", val_ptrs[gi], val_ptrs[gi - 1], vlen_const),
                    ],
                )

            # Assumes all initial indices are 0.
            emit_valu_chunked(
                octet_body,
                [("^", g["idx"], g["idx"], g["idx"]) for g in active_groups],
            )

            for gi in range(0, main_group_width, 2):
                emit_bundle(
                    octet_body,
                    load=[
                        ("vload", active_groups[gi]["val"], val_ptrs[gi]),
                        ("vload", active_groups[gi + 1]["val"], val_ptrs[gi + 1]),
                    ],
                )

            for bundle in build_octet_schedule(active_groups):
                emit_bundle(
                    octet_body,
                    alu=bundle.get("alu"),
                    valu=bundle["valu"],
                    load=bundle["load"],
                )

            for gi in range(0, main_group_width, 2):
                flow_ops = (
                    [("add_imm", loop_g_off, loop_g_off, main_group_width * VLEN)]
                    if gi == main_group_width - 2
                    else None
                )
                emit_bundle(
                    octet_body,
                    store=[
                        ("vstore", val_ptrs[gi], active_groups[gi]["val"]),
                        ("vstore", val_ptrs[gi + 1], active_groups[gi + 1]["val"]),
                    ],
                    flow=flow_ops,
                )

            octet_body_len = len(octet_body)
            self.instrs.extend(octet_body)
            self.instrs.append({"alu": [("<", cond, loop_g_off, loop_bound)]})
            self.instrs.append({"flow": [("cond_jump_rel", cond, -(octet_body_len + 2))]})

        processed_groups = octet_groups
        remaining_groups = n_groups - processed_groups
        tail_g_off = processed_groups * VLEN

        while remaining_groups >= 3:
            emit_tail_groups(3, tail_g_off)
            tail_g_off += three_vlen
            remaining_groups -= 3

        while remaining_groups >= 2:
            emit_tail_groups(2, tail_g_off)
            tail_g_off += two_vlen
            remaining_groups -= 2

        if remaining_groups == 1:
            emit_tail_groups(1, tail_g_off)

        # No pauses in the submission path; we validate against final memory.

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    machine.run()
    for ref_mem in reference_kernel2(mem, value_trace):
        pass

    inp_values_p = ref_mem[6]
    if prints:
        print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect output values"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
