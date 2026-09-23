"""A completed worker must not free a different worker's frozen CPU."""
import unittest
from experiments.pipeline.supervisor import available_run_index

class CampaignSchedulerTests(unittest.TestCase):
    def test_out_of_order_completions_preserve_cpu_exclusivity(self):
        remaining=[dict(id=f'{seed}-{cpu}',cpu=cpu) for seed in (0,1) for cpu in (4,5,16,17)]
        children={}
        for _ in range(4):
            run=remaining.pop(available_run_index(remaining,children))
            children[run['id']]=(None,run,None)
        self.assertIsNone(available_run_index(remaining,children))
        for cpu in (17,5,16,4):
            del children[f'0-{cpu}']
            run=remaining.pop(available_run_index(remaining,children))
            self.assertEqual(run['cpu'],cpu)
            children[run['id']]=(None,run,None)
            self.assertEqual(len({r['cpu'] for _,r,_ in children.values()}),4)
        self.assertIsNone(available_run_index(remaining,children))

    def test_fifo_for_unoccupied_cpus(self):
        self.assertEqual(available_run_index([{'cpu':4},{'cpu':5}],{}),0)
        self.assertIsNone(available_run_index([],{}))
