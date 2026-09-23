"""Literal scalar kernel expectations, timing, initialization and RNG isolation."""
import copy
import pickle
import unittest
import numpy as np
from config import SimConfig
from sim.plasticity import Plasticity
from sim.simulate import Simulation


def rule(mode='abbott_cascade', n=8, **kw):
    return Plasticity(1,n,1,5,.001,.009,mode=mode,transition_seed=17,**kw)


def literal(state, direction, mode, u, p_ltd=.9, p_ltp=.1):
    # Independent scalar reading of CbmSim's switch/case clauses, no imported tables.
    if direction == 0:
        return state
    if mode == 'binary':
        return (3 if direction < 0 else 4) if u < (p_ltd if direction < 0 else p_ltp) else state
    if direction < 0:
        if state == 0: return 0
        divisor = 2**(4-state) if state < 4 else 2**(state-4)
        return (state-1 if state < 4 or mode == 'mauk_cascade' else 3) if u < p_ltd/divisor else state
    if state == 7: return 7
    divisor = 2**(3-state) if state < 4 else 2**(state-3)
    return (state+1 if state > 3 or mode == 'mauk_cascade' else 4) if u < p_ltp/divisor else state


class TestCascadeTransitions(unittest.TestCase):
    def test_all_states_directions_and_probability_edges(self):
        for mode in ('abbott_cascade','mauk_cascade','binary'):
            for state in ((3,4) if mode=='binary' else range(8)):
                for direction in (-1,1):
                    threshold=(.9 if direction<0 else .1)
                    # Contains every transition threshold, exact edges and adjacent floats.
                    values=[0.,1.]
                    for divisor in (1,2,4,8):
                        p=threshold/divisor
                        values += [np.nextafter(p,0),p,np.nextafter(p,1)]
                    for u in values:
                        with self.subTest(mode=mode,state=state,direction=direction,u=u):
                            p=rule(mode,n=1);p.states[:]=state
                            w=np.full((1,1),.25 if state<4 else .55)
                            before=copy.deepcopy(p.transition_rng.bit_generator.state)
                            p.apply_resolved(w,[0],[direction],[u])
                            expected=literal(state,direction,mode,u)
                            self.assertEqual(int(p.states[0,0]),expected)
                            self.assertEqual(w[0,0],.25 if expected<4 else .55)
                            self.assertEqual(p.n_state_transitions,int(expected!=state))
                            self.assertEqual(p.n_weight_switches,int((expected<4)!=(state<4)))
                            self.assertEqual(p.n_ltd_events,int(direction<0))
                            self.assertEqual(p.n_ltp_events,int(direction>0))
                            self.assertEqual(before,p.transition_rng.bit_generator.state)
                            p.validate_state(w)

    def test_zero_and_one_probability(self):
        for mode in ('abbott_cascade','mauk_cascade','binary'):
            for probability in (0.,1.):
                p=rule(mode,n=2,p_ltd=probability,p_ltp=probability)
                p.states[:]=[4,3];w=np.array([[.55,.25]])
                p.apply_resolved(w,[0,1],[-1,1],[0.,0.])
                np.testing.assert_array_equal(p.states,[[3,4]] if probability else [[4,3]])

    def test_invalid_events_do_not_mutate(self):
        p=rule();w=np.full((1,8),.25)
        for ids,dirs,u in [([0,0],[-1,1],None),([-1],[1],None),([8],[1],None),([.5],[1],None),
                           ([0],[2],None),([0,1],[1],None),([0],[1],[np.nan]),([0],[1],[-.1]),([0],[1],[1.1])]:
            before=pickle.dumps(p)
            with self.assertRaises(ValueError):p.apply_resolved(w,ids,dirs,u)
            self.assertEqual(before,pickle.dumps(p))

    def test_null_and_empty_do_not_draw(self):
        p=rule();w=np.full((1,8),.25)
        before=copy.deepcopy(p.transition_rng.bit_generator.state)
        p.apply_resolved(w,np.arange(8),np.zeros(8));p.apply_resolved(w,[],[])
        self.assertEqual(before,p.transition_rng.bit_generator.state)
        self.assertEqual(p.n_null_events,8);self.assertEqual(p.n_state_transitions,0)
        np.testing.assert_array_equal(w,np.full((1,8),.25))

    def test_noncontiguous_weights(self):
        p=rule(n=2,p_ltp=1);base=np.full((1,4),.25);w=base[:,::2]
        p.apply_resolved(w,[0,1],[1,1],[0,0])
        np.testing.assert_array_equal(base,[[.55,.25,.55,.25]])

    def test_state_weight_corruption_rejected(self):
        p=rule();w=np.full((1,8),.5)
        with self.assertRaisesRegex(ValueError,'disagree'):p.apply_resolved(w,[0],[1])
        w[:]=.25;p.states[0,0]=8
        with self.assertRaisesRegex(ValueError,'states'):p.validate_state(w)


class TestCascadeIntegration(unittest.TestCase):
    def test_original_timing_and_null_boundaries(self):
        for mode in ('abbott_cascade','mauk_cascade','binary'):
            for null_steps in (0,3):
                for delay in (None,0,1,5,6,8,9):
                    p=rule(mode,n=1,null_window_ms=null_steps,p_ltd=1,p_ltp=1)
                    w=np.array([[.25]])
                    for t in range(15):
                        p.step(np.array([[t==0]]),delay==t,w)
                    expected_direction=(-1 if delay is not None and 1<=delay<=5 else
                                        0 if null_steps and delay is not None and 6<=delay<=8 else 1)
                    self.assertEqual(p.n_ltd_events,int(expected_direction<0))
                    self.assertEqual(p.n_ltp_events,int(expected_direction>0))
                    self.assertEqual(p.n_null_events,int(expected_direction==0))
                    self.assertEqual(w[0,0],.55 if expected_direction>0 else .25)

    def test_cf_ownership_and_no_activity(self):
        for mode in ('abbott_cascade','mauk_cascade','binary'):
            p=Plasticity(2,1,1,5,.001,.009,mode=mode,p_ltd=1,p_ltp=1,cf_source_of_pkj=[1,0])
            p.states[:]=4;w=np.full((2,1),.55)
            for t in range(10):p.step(np.full((2,1),t==0),[t==3,False],w)
            np.testing.assert_array_equal(w,[[.55],[.25]])
            before=pickle.dumps(p.transition_rng.bit_generator.state)
            for t in range(10):p.step(np.zeros((2,1),bool),[True,True],w)
            self.assertEqual(before,pickle.dumps(p.transition_rng.bit_generator.state))
            self.assertEqual(p.n_ltd_events,1);self.assertEqual(p.n_ltp_events,1)

    def test_initialization_and_rng_namespaces(self):
        configs=[SimConfig(seed=11,n_pf_per_pkj=5,plasticity_mode=mode,two_level_initialization=True)
                 for mode in ('additive','binary','abbott_cascade','mauk_cascade')]
        sims=[Simulation(c) for c in configs]
        for s in sims:
            np.testing.assert_array_equal(s.weights,sims[0].weights)
            self.assertEqual(s.rng.bit_generator.state,sims[0].rng.bit_generator.state)
            self.assertEqual(s.record_rng.bit_generator.state,sims[0].record_rng.bit_generator.state)
            np.testing.assert_array_equal(s.pkj.V,sims[0].pkj.V)
            np.testing.assert_array_equal(s.dcn.V,sims[0].dcn.V)
            np.testing.assert_array_equal(s.io.V,sims[0].io.V)
            s.plasticity.validate_state(s.weights)
        a=Simulation(SimConfig(seed=11,n_pf_per_pkj=5,plasticity_mode='abbott_cascade',weight_initialization_seed=987))
        self.assertFalse(np.array_equal(a.weights,sims[2].weights))
        self.assertEqual(a.plasticity.transition_rng.bit_generator.state,sims[2].plasticity.transition_rng.bit_generator.state)
        b=Simulation(SimConfig(seed=11,n_pf_per_pkj=5,plasticity_mode='abbott_cascade',plasticity_transition_seed=999))
        np.testing.assert_array_equal(b.weights,sims[2].weights)
        self.assertNotEqual(b.plasticity.transition_rng.bit_generator.state,sims[2].plasticity.transition_rng.bit_generator.state)
        default=Simulation(SimConfig(seed=11,n_pf_per_pkj=5))
        np.testing.assert_array_equal(default.weights,np.full(default.weights.shape,.5))
        self.assertFalse(hasattr(default.plasticity,'transition_rng'))
        self.assertEqual(default.rng.bit_generator.state,sims[0].rng.bit_generator.state)

    def test_invalid_model_settings(self):
        for kw in ({'mode':'bad'},{'weight_dependence':1},{'p_ltd':-.1},{'p_ltp':1.1},
                   {'p_ltp':float('nan')},{'low_weight':.6},{'high_weight':1.1}):
            with self.assertRaises(ValueError):
                Plasticity(1,1,1,5,.001,.009,**({'mode':'abbott_cascade'}|kw))
        with self.assertRaises(ValueError):Simulation(SimConfig(plasticity_mode='mauk_cascade',homeostatic_scaling=True))

    def test_default_additive_resolved_matches_literal_steps(self):
        for wd in (0.,.5,1.):
            p=rule('additive',n=5,weight_dependence=wd)
            w=np.array([[0.,.2,.5,.9,1.]])
            expected=w.copy();direction=np.array([-1,1,0,-1,1])
            down=(expected-0.)/1.;up=(1.-expected)/1.
            expected[0,direction==-1]-=.009*((1-wd)+wd*down[0,direction==-1])
            expected[0,direction==1]+=.001*((1-wd)+wd*up[0,direction==1])
            np.clip(expected,0,1,out=expected)
            p.apply_resolved(w,np.arange(5),direction)
            np.testing.assert_array_equal(w,expected)

    def test_exact_pickle_with_pending_eligibility_and_hidden_state(self):
        rng=np.random.default_rng(2)
        pf=rng.random((60,1,8))<.4;cf=rng.random(60)<.15
        for mode in ('abbott_cascade','mauk_cascade','binary'):
            p=rule(mode,null_window_ms=3);w=np.empty((1,8));p.initialize_weights(w,seed=19)
            for t in range(17):p.step(pf[t],cf[t],w)
            self.assertTrue(p._pf_ring.any())
            clone,wclone=pickle.loads(pickle.dumps((p,w),protocol=5))
            for t in range(17,60):
                p.step(pf[t],cf[t],w);clone.step(pf[t],cf[t],wclone)
            np.testing.assert_array_equal(w,wclone)
            self.assertEqual(pickle.dumps(p),pickle.dumps(clone))
            p.validate_state(w)

if __name__=='__main__':unittest.main()

class TestCascadeRecording(unittest.TestCase):
    def test_snapshot_schemas_and_state_counters(self):
        from experiments.pipeline.record import Recorder
        for mode in ('additive','binary','abbott_cascade','mauk_cascade'):
            sim=Simulation(SimConfig(seed=4,n_pf_per_pkj=5,plasticity_mode=mode))
            sim._choose_recorded();rec=Recorder(sim,2000,trace_seconds=0)
            if mode=='additive':
                self.assertNotIn('tracked_state',rec.schemas)
                continue
            ids=np.arange(sim.weights.size)
            # Every low synapse switches up; high states deepen for cascades.
            sim.plasticity.apply_resolved(sim.weights,ids,np.ones(len(ids)),np.zeros(len(ids)))
            rec.snapshot(sim,1000);d=rec.arrays()
            self.assertEqual(d['cascade_occupancy'].shape,(2,sim.conn.n_pkj,8))
            np.testing.assert_array_equal(d['cascade_occupancy'].sum(axis=2),np.full((2,sim.conn.n_pkj),5))
            np.testing.assert_array_equal(d['tracked_w'],np.where(d['tracked_state']<4,.25,.55))
            np.testing.assert_array_equal(d['cascade_counts'][-1],sim.plasticity.cascade_counters())
            self.assertTrue(np.all(np.diff(d['tracked_switch_count'],axis=0)>=0))
            sim.plasticity.validate_state(sim.weights)
