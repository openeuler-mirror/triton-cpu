// RUN: triton-shared-opt --split-input-file --triton-annotate-nontemporal-args %s | FileCheck %s

// The scalar %arg2 does not count. The argument that is never loaded (%arg0 here
// is only stored to, temporally) is not listed for either kind.
module {
  tt.func public @only_evict_first(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.load %2 evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %5, %3 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @only_evict_first
// CHECK-SAME:  attributes {tts.nontemporal_load_args = array<i32: 1>, tts.ptr_arg_count = 2 : i32}
// CHECK-NOT:   tts.nontemporal_store_args

// -----

// The same buffer read both ways: the hint is ambiguous per argument, so the
// argument is not listed.
module {
  tt.func public @mixed(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c128 = arith.constant 128 : i32
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.load %2 evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
    %4 = tt.addptr %arg0, %c128 : !tt.ptr<f32>, i32
    %5 = tt.splat %4 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %6 = tt.addptr %5, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %7 = tt.load %6 : tensor<128x!tt.ptr<f32>>
    %8 = arith.addf %3, %7 : tensor<128xf32>
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %10 = tt.addptr %9, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %10, %8 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @mixed
// CHECK-NOT:   tts.nontemporal_load_args

// -----

// one buffer passed twice, program 0 streams through one argument and everyone else reads the
// other temporally, inside scf.if. Only the streaming argument is listed.
module {
  tt.func public @two_policies_two_args(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.cmpi eq, %0, %c0 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = scf.if %1 -> (tensor<128xf32>) {
      %s = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %p = tt.addptr %s, %2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %v = tt.load %p evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
      scf.yield %v : tensor<128xf32>
    } else {
      %s = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %p = tt.addptr %s, %2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      %v = tt.load %p : tensor<128x!tt.ptr<f32>>
      scf.yield %v : tensor<128xf32>
    }
    %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %2 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %5, %3 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @two_policies_two_args
// CHECK-SAME:  tts.nontemporal_load_args = array<i32: 1>, tts.ptr_arg_count = 3 : i32

// -----

// A pointer carried through scf.for iter_args still roots at its argument:
// both the init operand and the yielded value are followed.
module {
  tt.func public @loop_carried(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c128 = arith.constant 128 : i32
    %cst = arith.constant dense<0.0> : tensor<128xf32>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.splat %c128 : i32 -> tensor<128xi32>
    %4:2 = scf.for %i = %c0 to %c4 step %c4 iter_args(%p = %2, %acc = %cst) -> (tensor<128x!tt.ptr<f32>>, tensor<128xf32>) : i32 {
      %v = tt.load %p evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
      %a = arith.addf %acc, %v : tensor<128xf32>
      %n = tt.addptr %p, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      scf.yield %n, %a : tensor<128x!tt.ptr<f32>>, tensor<128xf32>
    }
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %6 = tt.addptr %5, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %6, %4#1 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @loop_carried
// CHECK-SAME:  tts.nontemporal_load_args = array<i32: 0>, tts.ptr_arg_count = 2 : i32

// -----

// evict_last is not evict_first: nothing to record.
module {
  tt.func public @evict_last_only(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.load %2 evictionPolicy = evict_last : tensor<128x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %5, %3 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @evict_last_only
// CHECK-NOT:   tts.nontemporal_load_args

// -----

// Stores carry the hint too, and are judged on their own: %arg1 is only
// written with evict_first (listed for stores), %arg0 is only read
// temporally (listed for neither).
module {
  tt.func public @store_evict_first(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.load %2 : tensor<128x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %5, %3 evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @store_evict_first
// CHECK-SAME:  attributes {tts.nontemporal_store_args = array<i32: 1>, tts.ptr_arg_count = 2 : i32}
// CHECK-NOT:   tts.nontemporal_load_args

// -----

// One argument, both kinds, each with its own policy: read temporally,
// written with evict_first.  Listed for stores only -- the kinds do not
// contaminate each other.
module {
  tt.func public @load_temporal_store_evict(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.load %2 : tensor<128x!tt.ptr<f32>>
    %4 = arith.addf %3, %3 : tensor<128xf32>
    tt.store %2, %4 evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @load_temporal_store_evict
// CHECK-SAME:  attributes {tts.nontemporal_store_args = array<i32: 0>, tts.ptr_arg_count = 1 : i32}
// CHECK-NOT:   tts.nontemporal_load_args

// -----

// Both kinds evict_first on the same argument: listed for both.
module {
  tt.func public @both_kinds(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.load %2 evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
    tt.store %2, %3 evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @both_kinds
// CHECK-SAME:  attributes {tts.nontemporal_load_args = array<i32: 0>, tts.nontemporal_store_args = array<i32: 0>, tts.ptr_arg_count = 1 : i32}

// -----

// A store argument written both ways is ambiguous, exactly like a load one.
module {
  tt.func public @mixed_store(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c128 = arith.constant 128 : i32
    %cst = arith.constant dense<1.0> : tensor<128xf32>
    %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %2, %cst evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
    %3 = tt.addptr %arg1, %c128 : !tt.ptr<f32>, i32
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %5, %cst : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @mixed_store
// CHECK-NOT:   tts.nontemporal

// -----

// A masked evict_first load must not be listed.  AArch64 has no pattern for an
// access that is both masked and non-temporal once vscale_range widens it to a
// scalable vector -- llc aborts with
//   Cannot select: masked_store<(non-temporal store ...)>
// -- and the hint is per pointer argument, so one masked access disqualifies
// the argument for that kind.
module {
  tt.func public @masked_evict_first_load(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %cst = arith.constant dense<0.0> : tensor<128xf32>
    %1 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %mask = arith.cmpi slt, %0, %1 : tensor<128xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %3 = tt.addptr %2, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %4 = tt.load %3, %mask, %cst evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %6 = tt.addptr %5, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %6, %4 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @masked_evict_first_load
// CHECK-NOT:   tts.nontemporal

// -----

// Same for stores -- this is the shape flag_gems' exponential_ kernel uses,
// tl.store(..., mask=off < N, eviction_policy="evict_first"), which is what
// made llc abort.
module {
  tt.func public @masked_evict_first_store(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %cst = arith.constant dense<1.0> : tensor<128xf32>
    %1 = tt.splat %arg1 : i32 -> tensor<128xi32>
    %mask = arith.cmpi slt, %0, %1 : tensor<128xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %3 = tt.addptr %2, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %3, %cst, %mask evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @masked_evict_first_store
// CHECK-NOT:   tts.nontemporal

// -----

// One argument, an unmasked evict_first store and a masked one: the masked
// access disqualifies the argument, because the attribute cannot distinguish
// the two accesses.
module {
  tt.func public @mixed_masked_and_plain(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %cst = arith.constant dense<1.0> : tensor<128xf32>
    %c128 = arith.constant 128 : i32
    %1 = tt.splat %arg1 : i32 -> tensor<128xi32>
    %mask = arith.cmpi slt, %0, %1 : tensor<128xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %3 = tt.addptr %2, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %3, %cst evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
    %4 = tt.addptr %arg0, %c128 : !tt.ptr<f32>, i32
    %5 = tt.splat %4 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %6 = tt.addptr %5, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %6, %cst, %mask evictionPolicy = evict_first : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: tt.func public @mixed_masked_and_plain
// CHECK-NOT:   tts.nontemporal
