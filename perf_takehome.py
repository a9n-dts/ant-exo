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
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Vector work registers for two interleaved groups (VLEN=8 words each)
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

        # Vector constants (broadcast from scalars)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_fvp = self.alloc_scratch("v_fvp", VLEN)

        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", v_fvp, self.scratch["forest_values_p"]))

        # Broadcast hash stage constants to vectors
        v_hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1 = self.alloc_scratch(f"v_h{hi}_c1", VLEN)
            v_c3 = self.alloc_scratch(f"v_h{hi}_c3", VLEN)
            s_c1 = self.scratch_const(val1)
            s_c3 = self.scratch_const(val3)
            self.add("valu", ("vbroadcast", v_c1, s_c1))
            self.add("valu", ("vbroadcast", v_c3, s_c3))
            v_hash_consts.append((v_c1, v_c3))

        # Loop variables
        loop_g_off = self.alloc_scratch("loop_g_off")
        cond = self.alloc_scratch("cond")
        vlen_const = self.scratch_const(VLEN)
        two_vlen = 2 * VLEN

        self.add("flow", ("pause",))

        # Process two vector groups per iteration to overlap gather(load) with hash(valu).
        paired_groups = (n_groups // 2) * 2
        if paired_groups > 0:
            self.add("load", ("const", loop_g_off, 0))
            paired_batch = paired_groups * VLEN
            loop_bound = (
                self.scratch["batch_size"]
                if paired_batch == batch_size
                else self.scratch_const(paired_batch)
            )

            pair_body = []
            pair_body.append(("alu", ("+", tmp1, self.scratch["inp_indices_p"], loop_g_off)))
            pair_body.append(("alu", ("+", tmp2, self.scratch["inp_values_p"], loop_g_off)))
            pair_body.append(("alu", ("+", tmp3, tmp1, vlen_const)))
            pair_body.append(("alu", ("+", tmp4, tmp2, vlen_const)))
            pair_body.append(("load", ("vload", v_idx_a, tmp1)))
            pair_body.append(("load", ("vload", v_val_a, tmp2)))
            pair_body.append(("load", ("vload", v_idx_b, tmp3)))
            pair_body.append(("load", ("vload", v_val_b, tmp4)))

            for _ in range(rounds):
                pair_body.append(("valu", ("+", v_addr_a, v_fvp, v_idx_a)))
                pair_body.append(("valu", ("+", v_addr_b, v_fvp, v_idx_b)))
                for offset in range(VLEN):
                    pair_body.append(("load", ("load_offset", v_node_val_a, v_addr_a, offset)))

                pair_body.append(("valu", ("^", v_val_a, v_val_a, v_node_val_a)))

                b_load_offset = 0
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    vc1, vc3 = v_hash_consts[hi]
                    pair_body.append(("valu", (op1, v_tmp1_a, v_val_a, vc1)))
                    pair_body.append(("valu", (op3, v_tmp2_a, v_val_a, vc3)))
                    if b_load_offset < VLEN:
                        pair_body.append(("load", ("load_offset", v_node_val_b, v_addr_b, b_load_offset)))
                        b_load_offset += 1
                    if b_load_offset < VLEN:
                        pair_body.append(("load", ("load_offset", v_node_val_b, v_addr_b, b_load_offset)))
                        b_load_offset += 1
                    pair_body.append(("valu", (op2, v_val_a, v_tmp1_a, v_tmp2_a)))

                pair_body.append(("valu", ("&", v_tmp1_a, v_val_a, v_one)))
                pair_body.append(("valu", ("multiply_add", v_idx_a, v_idx_a, v_two, v_one)))
                pair_body.append(("valu", ("+", v_idx_a, v_idx_a, v_tmp1_a)))
                pair_body.append(("valu", ("<", v_tmp1_a, v_idx_a, v_n_nodes)))
                pair_body.append(("flow", ("vselect", v_idx_a, v_tmp1_a, v_idx_a, v_zero)))

                pair_body.append(("valu", ("^", v_val_b, v_val_b, v_node_val_b)))
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    vc1, vc3 = v_hash_consts[hi]
                    pair_body.append(("valu", (op1, v_tmp1_b, v_val_b, vc1)))
                    pair_body.append(("valu", (op3, v_tmp2_b, v_val_b, vc3)))
                    pair_body.append(("valu", (op2, v_val_b, v_tmp1_b, v_tmp2_b)))

                pair_body.append(("valu", ("&", v_tmp1_b, v_val_b, v_one)))
                pair_body.append(("valu", ("multiply_add", v_idx_b, v_idx_b, v_two, v_one)))
                pair_body.append(("valu", ("+", v_idx_b, v_idx_b, v_tmp1_b)))
                pair_body.append(("valu", ("<", v_tmp1_b, v_idx_b, v_n_nodes)))
                pair_body.append(("flow", ("vselect", v_idx_b, v_tmp1_b, v_idx_b, v_zero)))

            pair_body.append(("store", ("vstore", tmp1, v_idx_a)))
            pair_body.append(("store", ("vstore", tmp2, v_val_a)))
            pair_body.append(("store", ("vstore", tmp3, v_idx_b)))
            pair_body.append(("store", ("vstore", tmp4, v_val_b)))
            pair_body.append(("flow", ("add_imm", loop_g_off, loop_g_off, two_vlen)))

            pair_body_instrs = self.build(pair_body)
            pair_body_len = len(pair_body_instrs)
            self.instrs.extend(pair_body_instrs)
            self.instrs.append({"alu": [("<", cond, loop_g_off, loop_bound)]})
            self.instrs.append({"flow": [("cond_jump_rel", cond, -(pair_body_len + 2))]})

        # Tail for odd number of groups (if any): process the final single group.
        if n_groups % 2 == 1:
            tail_g_off = self.scratch_const(paired_groups * VLEN)
            tail_body = []
            tail_body.append(("alu", ("+", tmp1, self.scratch["inp_indices_p"], tail_g_off)))
            tail_body.append(("alu", ("+", tmp2, self.scratch["inp_values_p"], tail_g_off)))
            tail_body.append(("load", ("vload", v_idx_a, tmp1)))
            tail_body.append(("load", ("vload", v_val_a, tmp2)))

            for _ in range(rounds):
                tail_body.append(("valu", ("+", v_addr_a, v_fvp, v_idx_a)))
                for offset in range(VLEN):
                    tail_body.append(("load", ("load_offset", v_node_val_a, v_addr_a, offset)))
                tail_body.append(("valu", ("^", v_val_a, v_val_a, v_node_val_a)))
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    vc1, vc3 = v_hash_consts[hi]
                    tail_body.append(("valu", (op1, v_tmp1_a, v_val_a, vc1)))
                    tail_body.append(("valu", (op3, v_tmp2_a, v_val_a, vc3)))
                    tail_body.append(("valu", (op2, v_val_a, v_tmp1_a, v_tmp2_a)))
                tail_body.append(("valu", ("&", v_tmp1_a, v_val_a, v_one)))
                tail_body.append(("valu", ("multiply_add", v_idx_a, v_idx_a, v_two, v_one)))
                tail_body.append(("valu", ("+", v_idx_a, v_idx_a, v_tmp1_a)))
                tail_body.append(("valu", ("<", v_tmp1_a, v_idx_a, v_n_nodes)))
                tail_body.append(("flow", ("vselect", v_idx_a, v_tmp1_a, v_idx_a, v_zero)))

            tail_body.append(("store", ("vstore", tmp1, v_idx_a)))
            tail_body.append(("store", ("vstore", tmp2, v_val_a)))
            self.instrs.extend(self.build(tail_body))

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
