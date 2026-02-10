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

        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        tmp4 = self.alloc_scratch("tmp4")
        tmp5 = self.alloc_scratch("tmp5")
        tmp6 = self.alloc_scratch("tmp6")

        # Load init vars from memory header
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Scalar constants
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        tmp7 = self.alloc_scratch("tmp7")
        tmp8 = self.alloc_scratch("tmp8")
        tmp9 = self.alloc_scratch("tmp9")
        tmp10 = self.alloc_scratch("tmp10")
        tmp11 = self.alloc_scratch("tmp11")
        tmp12 = self.alloc_scratch("tmp12")

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

        # Vector constants (broadcast from scalars)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_fvp = self.alloc_scratch("v_fvp", VLEN)
        v_root = self.alloc_scratch("v_root", VLEN)
        root_node = self.alloc_scratch("root_node")

        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_fvp, self.scratch["forest_values_p"]))
        self.add("load", ("load", root_node, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", v_root, root_node))

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
        six_vlen = 6 * VLEN

        val_ptrs = [tmp2, tmp4, tmp6, tmp8, tmp10, tmp12]

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
        ]
        triplet_a = groups_all[:3]
        triplet_b = groups_all[3:6]
        wrap_period = forest_height + 1

        def append_round_ops_inphase(body, groups, round_idx):
            if round_idx % wrap_period == 0:
                for g in groups:
                    body.append(("valu", ("^", g["val"], g["val"], v_root)))
            else:
                for g in groups:
                    body.append(("valu", ("+", g["addr"], v_fvp, g["idx"])))
                for offset in range(VLEN):
                    for g in groups:
                        body.append(("load", ("load_offset", g["node"], g["addr"], offset)))
                for g in groups:
                    body.append(("valu", ("^", g["val"], g["val"], g["node"])))

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

        def hash_cycles(groups, round_idx, need_next_addr):
            cycles = []
            for stage in hash_stage_plan:
                if stage[0] == "muladd":
                    _, v_mul, v_add = stage
                    cycles.append(
                        [
                            ("multiply_add", g["val"], g["val"], v_mul, v_add)
                            for g in groups
                        ]
                    )
                else:
                    _, op1, op2, op3, vc1, vc3 = stage
                    cycles.append(
                        [(op1, g["tmp1"], g["val"], vc1) for g in groups]
                        + [(op3, g["tmp2"], g["val"], vc3) for g in groups]
                    )
                    cycles.append(
                        [(op2, g["val"], g["tmp1"], g["tmp2"]) for g in groups]
                    )

            if (round_idx + 1) % wrap_period == 0:
                cycles.append([("^", g["idx"], g["idx"], g["idx"]) for g in groups])
            else:
                cycles.append(
                    [("&", g["tmp1"], g["val"], v_one) for g in groups]
                    + [("multiply_add", g["idx"], g["idx"], v_two, v_one) for g in groups]
                )
                cycles.append([("+", g["idx"], g["idx"], g["tmp1"]) for g in groups])

            if need_next_addr:
                cycles.append([("+", g["addr"], v_fvp, g["idx"]) for g in groups])
            return cycles

        def emit_gather_only(body, groups, gather_round_idx):
            if gather_round_idx % wrap_period == 0:
                emit_bundle(body, valu=[("^", g["val"], g["val"], v_root) for g in groups])
                return
            load_ops = []
            for offset in range(VLEN):
                for g in groups:
                    load_ops.append(("load_offset", g["node"], g["addr"], offset))

            for ci in range(12):
                emit_bundle(body, load=load_ops[2 * ci : 2 * ci + 2])
            emit_bundle(body, valu=[("^", g["val"], g["val"], g["node"]) for g in groups])

        def emit_overlap_phase(
            body, gather_groups, hash_groups, round_idx, need_next_addr, gather_round_idx
        ):
            hcycles = hash_cycles(hash_groups, round_idx, need_next_addr)
            gather_is_root = gather_round_idx % wrap_period == 0
            phase_len = len(hcycles) if gather_is_root else max(len(hcycles), 13)
            bundles = [{"valu": [], "load": []} for _ in range(phase_len)]

            for ci, ops in enumerate(hcycles):
                bundles[ci]["valu"].extend(ops)

            if not gather_is_root:
                load_ops = []
                for offset in range(VLEN):
                    for g in gather_groups:
                        load_ops.append(("load_offset", g["node"], g["addr"], offset))
                for ci in range(12):
                    bundles[ci]["load"].extend(load_ops[2 * ci : 2 * ci + 2])

            xor_src = v_root if gather_is_root else None
            xor_ops = []
            for g in gather_groups:
                if gather_is_root:
                    xor_ops.append(("^", g["val"], g["val"], xor_src))
                else:
                    xor_ops.append(("^", g["val"], g["val"], g["node"]))
            xor_cycle = None
            start_ci = 0 if gather_is_root else 12
            for ci in range(start_ci, phase_len):
                if len(bundles[ci]["valu"]) <= 3:
                    xor_cycle = ci
                    break
            if xor_cycle is None:
                bundles.append({"valu": xor_ops, "load": []})
            else:
                bundles[xor_cycle]["valu"].extend(xor_ops)

            for b in bundles:
                emit_bundle(body, valu=b["valu"], load=b["load"])

        def emit_hash_only(body, groups, round_idx, need_next_addr):
            for ops in hash_cycles(groups, round_idx, need_next_addr):
                emit_bundle(body, valu=ops)

        def emit_tail_groups(group_count, g_off_words):
            groups = groups_all[:group_count]
            g_off_const = self.scratch_const(g_off_words)
            tail_body = []

            tail_body.append(
                ("alu", ("+", val_ptrs[0], self.scratch["inp_values_p"], g_off_const))
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

        self.add("flow", ("pause",))

        # Software-pipelined sextet loop: overlap gather(load) of one triplet
        # with hash/index(valu) of the other triplet.
        sextet_groups = (n_groups // 6) * 6
        if sextet_groups > 0:
            self.add("load", ("const", loop_g_off, 0))
            sextet_batch = sextet_groups * VLEN
            loop_bound = (
                self.scratch["batch_size"]
                if sextet_batch == batch_size
                else self.scratch_const(sextet_batch)
            )

            sextet_body = []

            emit_bundle(
                sextet_body,
                alu=[
                    ("+", val_ptrs[0], self.scratch["inp_values_p"], loop_g_off),
                ],
            )
            for gi in range(1, 6):
                emit_bundle(
                    sextet_body,
                    alu=[
                        ("+", val_ptrs[gi], val_ptrs[gi - 1], vlen_const),
                    ],
                )

            # Assumes all initial indices are 0.
            emit_bundle(
                sextet_body,
                valu=[("^", g["idx"], g["idx"], g["idx"]) for g in groups_all],
            )

            for gi, g in enumerate(groups_all):
                emit_bundle(
                    sextet_body,
                    load=[
                        ("vload", g["val"], val_ptrs[gi]),
                    ],
                )

            # Initial round-0 addresses for both triplets.
            emit_bundle(
                sextet_body,
                valu=[("+", g["addr"], v_fvp, g["idx"]) for g in groups_all],
            )

            # Prologue: gather triplet A round 0 so hash(A,0) can start.
            emit_gather_only(sextet_body, triplet_a, 0)

            for round_idx in range(rounds):
                need_a_next = round_idx < rounds - 1
                emit_overlap_phase(
                    sextet_body,
                    gather_groups=triplet_b,
                    hash_groups=triplet_a,
                    round_idx=round_idx,
                    need_next_addr=need_a_next,
                    gather_round_idx=round_idx,
                )
                if round_idx < rounds - 1:
                    emit_overlap_phase(
                        sextet_body,
                        gather_groups=triplet_a,
                        hash_groups=triplet_b,
                        round_idx=round_idx,
                        need_next_addr=True,
                        gather_round_idx=round_idx + 1,
                    )
                else:
                    emit_hash_only(
                        sextet_body,
                        groups=triplet_b,
                        round_idx=round_idx,
                        need_next_addr=False,
                    )

            for gi, g in enumerate(groups_all):
                flow_ops = [("add_imm", loop_g_off, loop_g_off, six_vlen)] if gi == 5 else None
                emit_bundle(
                    sextet_body,
                    store=[
                        ("vstore", val_ptrs[gi], g["val"]),
                    ],
                    flow=flow_ops,
                )

            sextet_body_len = len(sextet_body)
            self.instrs.extend(sextet_body)
            self.instrs.append({"alu": [("<", cond, loop_g_off, loop_bound)]})
            self.instrs.append({"flow": [("cond_jump_rel", cond, -(sextet_body_len + 2))]})

        processed_groups = sextet_groups
        remaining_groups = n_groups - processed_groups
        tail_g_off = processed_groups * VLEN

        if remaining_groups >= 3:
            emit_tail_groups(3, tail_g_off)
            tail_g_off += three_vlen
            remaining_groups -= 3

        if remaining_groups >= 2:
            emit_tail_groups(2, tail_g_off)
            tail_g_off += two_vlen
            remaining_groups -= 2

        if remaining_groups == 1:
            emit_tail_groups(1, tail_g_off)

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

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
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

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
