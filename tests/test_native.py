"""Exact scientific equivalence for the optional compiled backend."""
import copy
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from config import IOChannelParams, SimConfig
from sim.io_channels import IOPopulation
from sim.simulate import Simulation
from sim.poisson_input import generate_pf_spikes
from sim.native import advance_io, bernoulli
from experiments.pipeline.common import exact_equal

@unittest.skipIf(advance_io is None, 'Build optional native backend with python3 -m sim.build_native')
class NativeEquivalenceTests(unittest.TestCase):
    def test_io_state_events_and_rng(self):
        for n in (1, 7, 40, 83):
            for cv in (0., .15):
                p=IOChannelParams(noise_sigma=.7)
                rng=np.random.default_rng(77)
                gap=rng.uniform(0,.003,(n,n));gap=(gap+gap.T)/2;np.fill_diagonal(gap,0.)
                a=IOPopulation(p,n,1.,rng=np.random.default_rng(2),g_gap=gap,heterogeneity_cv=cv)
                b=copy.deepcopy(a)
                for step in range(80):
                    gg=rng.uniform(0,.4,n)
                    with patch('sim.io_channels.advance_io',None): fa=a.step(1.,gg)
                    fb=b.step(1.,gg)
                    np.testing.assert_array_equal(fa,fb)
                exact_equal(a,b)

    def test_singular_clamps_gap_off_and_parameter_swap(self):
        p=IOChannelParams()
        a=IOPopulation(p,7,.7,rng=np.random.default_rng(4));b=copy.deepcopy(a)
        initial=np.array([-300.,-200.,-8.5000001,-8.5,-8.4999999,100.,200.])
        a.V=initial.copy();b.V=initial.copy()
        for step in range(100):
            if step==30:
                newer=copy.deepcopy(p);newer.i_app=.2;newer.noise_sigma=.3
                a.set_params(copy.deepcopy(newer));b.set_params(copy.deepcopy(newer))
            with patch('sim.io_channels.advance_io',None): fa=a.step(.7,.05)
            fb=b.step(.7,.05)
            np.testing.assert_array_equal(fa,fb)
            exact_equal(a,b)

    def test_pcg64_states_shapes_probabilities_and_uint32_cache(self):
        states=[0,1,(1<<64)-1,1<<64,(1<<127)-1,(1<<128)-1]
        rng=np.random.default_rng(421)
        states += [int.from_bytes(rng.bytes(16),'little') for _ in range(30)]
        for initial in states:
            a=np.random.default_rng(2)
            state=a.bit_generator.state
            state['state']['state']=initial
            state['state']['inc']=int.from_bytes(rng.bytes(16),'little')|1
            state['has_uint32']=1;state['uinteger']=1234567
            a.bit_generator.state=state;b=copy.deepcopy(a)
            for shape in ((),(0,),(1,),(2,),(3,),(4,),(5,),(7,9),(320,500)):
                for probability in (0.,.02,.5,1.,np.nan):
                    expected=a.random(shape)<probability
                    actual=generate_pf_spikes(probability*1000,1.,b,shape)
                    self.assertIs(type(actual),type(expected))
                    np.testing.assert_array_equal(expected,actual)
                    exact_equal(a,b)
            np.testing.assert_array_equal(a.integers(0,2**32,13,dtype=np.uint32),b.integers(0,2**32,13,dtype=np.uint32))

    def test_other_bitgenerators_keep_reference(self):
        for cls in (np.random.Philox,np.random.SFC64,np.random.MT19937,np.random.PCG64DXSM):
            a=np.random.Generator(cls(44));b=copy.deepcopy(a)
            np.testing.assert_array_equal(a.random((17,13))<.02,generate_pf_spikes(20.,1.,b,(17,13)))
            exact_equal(a,b)

    def test_all_four_network_modes_complete_state(self):
        for mode in ('additive','binary','abbott_cascade','mauk_cascade'):
            cfg=SimConfig(seed=1,plasticity_mode=mode,two_level_initialization=True)
            a=Simulation(cfg);b=copy.deepcopy(a)
            for step in range(300):
                with patch('sim.io_channels.advance_io',None),patch('sim.poisson_input.bernoulli',None),patch('sim.plasticity.discrete_update',None): fa=a._step()
                fb=b._step()
                for x,y in zip(fa,fb): np.testing.assert_array_equal(x,y)
                a.t_ms+=cfg.dt_ms;b.t_ms+=cfg.dt_ms
            exact_equal(a,b)

    def test_missing_or_corrupt_build_never_silently_falls_back(self):
        from sim import native
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(native,'HERE',root):
                self.assertEqual(native._load()[3]['backend'],'python_reference')
                (root/'_io_native_fake.so').write_bytes(b'not a binary')
                with self.assertRaises(RuntimeError):native._load()
                (root/'native_build.json').write_text(json.dumps(native.BACKEND_IDENTITY['build']))
                with self.assertRaises(RuntimeError):native._load()

    def test_external_state_array_reference_semantics(self):
        a=IOPopulation(IOChannelParams(),3,1.,rng=np.random.default_rng(2));b=copy.deepcopy(a)
        old_a={k:getattr(a,k) for k in ('V','Ca','noise','k','l','r','q','s')}
        old_b={k:getattr(b,k) for k in old_a}
        with patch('sim.io_channels.advance_io',None):a.step(1.)
        b.step(1.)
        exact_equal(old_a,old_b)
        for key in old_a:self.assertEqual(old_a[key] is getattr(a,key),old_b[key] is getattr(b,key))

if __name__=='__main__':unittest.main()
