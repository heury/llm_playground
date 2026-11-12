import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/review.py'
spec = importlib.util.spec_from_file_location('review', SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'notebook.py').write_text('print(1)\n')
        (self.root/'model.py').write_text('def model(x):\n    return x + 1\n')
        self.contract = json.loads((SCRIPT.parents[1]/'assets/contract-template.json').read_text())
        self.contract.update(template_only=False, contract_id='fixture', project='test', run_id='fixture-v1',
                             learner='Python beginner', objectives=['explain a model'], required_theory=['addition'],
                             environment='CPU', target_files=[self.file('notebook.py')],
                             reference_implementations=[dict(self.file('model.py'), symbols=['model'], reason='actual import')],
                             projects_comparison={'applicable':False,'reason':'standalone model'},
                             units=[{'id':'intro','section':'Introduction','concepts':['addition'],'symbols':['model']}])
        for cp in self.contract['checkpoints']:
            cp['units'] = ['intro']
            cp['critical'] = cp['id'] == 'S2.1'
        self.result = {'contract_id':'fixture','contract_version':1,'rubric_version':'2.0',
                       'role':'student','phase':'baseline','target_hashes':{'notebook.py':self.file('notebook.py')['sha256']},
                       'references_verified':True,'findings':[], 'checkpoints':[
                           {'id':id,'units':[{'id':'intro','status':'pass','evidence':'notebook.py:1 fixture evidence'}]}
                           for id in sorted(review.expected_ids('student'))]}

    def file(self, name):
        return {'path':name, 'sha256':hashlib.sha256((self.root/name).read_bytes()).hexdigest()}

    def change(self, id, status):
        next(c for c in self.result['checkpoints'] if c['id']==id)['units'][0]['status']=status

    def scored(self):
        return review.score(self.contract,self.result,self.root)

    def test_complete_pass(self):
        s=self.scored()
        self.assertEqual((s['observed_score'],s['lower_bound'],s['verdict']),(100,100,'use'))
        self.assertFalse(s['execution_verified'])

    def test_unknown_inflation_does_not_allow_use(self):
        for i in range(1,5):self.change(f'S5.{i}','unknown')
        s=self.scored()
        self.assertEqual((s['observed_score'],s['coverage'],s['lower_bound'],s['verdict']),(100,.8,80,'hold'))

    def test_lower_bound_85(self):
        for i in range(2,5):self.change(f'S5.{i}','unknown')
        self.assertEqual(self.scored()['verdict'],'use')

    def test_critical_unknown_at_high_score(self):
        self.change('S2.1','unknown')
        self.assertEqual(self.scored()['verdict'],'hold')

    def test_failed_critical_without_finding(self):
        self.change('S2.1','fail')
        self.assertEqual(self.scored()['verdict'],'revise')

    def test_major_defect_at_95(self):
        self.change('S5.4','fail')
        self.result['findings']=[{'id':'S-01','severity':'major','status':'partial','checkpoints':['S5.4'],'evidence':'not fixed'}]
        self.assertEqual(self.scored()['verdict'],'revise')

    def test_missing_references(self):
        self.result['references_verified']=False
        self.assertEqual(self.scored()['verdict'],'hold')

    def test_missing_and_duplicate_observations(self):
        self.result['checkpoints'][0]['units']=[]
        with self.assertRaisesRegex(ValueError,'observations'):self.scored()
        self.result['checkpoints'][0]['units']=[{'id':'intro','status':'pass','evidence':'ok'}]*2
        with self.assertRaisesRegex(ValueError,'duplicate'):self.scored()

    def test_some_sections_fail(self):
        self.contract['units'].append({'id':'second','section':'Second','concepts':[],'symbols':[]})
        cp=next(c for c in self.contract['checkpoints'] if c['id']=='S2.1')
        cp['units'].append('second')
        row=next(c for c in self.result['checkpoints'] if c['id']=='S2.1')
        row['units'].append({'id':'second','status':'fail','evidence':'shape omitted'})
        s=self.scored()
        self.assertEqual((s['checkpoint_states']['S2.1'],s['verdict']),('fail','revise'))

    def test_exclusion_cannot_hide_critical(self):
        cp=next(c for c in self.contract['checkpoints'] if c['id']=='S2.1')
        cp.update(units=[],excluded_reason='not applicable')
        with self.assertRaisesRegex(ValueError,'exclusion'):self.scored()

    def test_noncritical_exclusion(self):
        cp=next(c for c in self.contract['checkpoints'] if c['id']=='S5.1')
        cp.update(units=[],excluded_reason='outside scope')
        next(c for c in self.result['checkpoints'] if c['id']=='S5.1')['units']=[]
        self.assertEqual(self.scored()['lower_bound'],100)

    def test_template_and_wrong_version_rejected(self):
        self.contract['template_only']=True
        with self.assertRaisesRegex(ValueError,'template'):self.scored()
        self.contract['template_only']=False
        self.result['contract_version']=2
        with self.assertRaisesRegex(ValueError,'mismatch'):self.scored()

    def test_stale_baseline_source(self):
        (self.root/'notebook.py').write_text('print(2)\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.scored()

    def test_after_phase(self):
        (self.root/'notebook.py').write_text('print(2)\n')
        self.contract['after_files']=[self.file('notebook.py')]
        self.result.update(phase='after',target_hashes={'notebook.py':self.file('notebook.py')['sha256']})
        self.assertEqual(self.scored()['verdict'],'use')

    def test_reference_drift_rejected(self):
        (self.root/'model.py').write_text('def model(x):\n    return x + 2\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.scored()

    def test_update_requires_validation_plan(self):
        self.contract['execution_scope']['source_edits']=True
        with self.assertRaisesRegex(ValueError,'validation plan'):self.scored()

    def test_embedded_reference_survives_prose_only_edit(self):
        (self.root/'notebook.py').write_text('def model(x):\n    return x + 1\n\nprint("old prose")\n')
        self.contract['target_files']=[self.file('notebook.py')]
        ast_hash=review.embedded_fingerprint(self.root/'notebook.py',['model'])
        self.contract['reference_implementations']=[{'kind':'embedded','path':'notebook.py','sha256':ast_hash,'symbols':['model'],'reason':'inline code'}]
        (self.root/'notebook.py').write_text('def model(x):\n    return x + 1\n\nprint("new prose")\n')
        self.contract['after_files']=[self.file('notebook.py')]
        self.result.update(phase='after',target_hashes={'notebook.py':self.file('notebook.py')['sha256']})
        self.assertEqual(self.scored()['verdict'],'use')
        (self.root/'notebook.py').write_text('def model(x):\n    return x + 2\n')
        self.contract['after_files']=[self.file('notebook.py')]
        self.result['target_hashes']={'notebook.py':self.file('notebook.py')['sha256']}
        with self.assertRaisesRegex(ValueError,'implementation changed'):self.scored()

    def test_all_unknown(self):
        for row in self.result['checkpoints']:
            row['units'][0]['status']='unknown'
        s=self.scored()
        self.assertEqual((s['observed_score'],s['coverage'],s['lower_bound'],s['upper_bound'],s['verdict']),
                         (None,0,0,100,'hold'))

    def test_all_role_checkpoints_excluded(self):
        for cp in self.contract['checkpoints']:
            if cp['role']=='student':
                cp.update(units=[],excluded_reason='out of role scope',critical=False)
            elif cp['id']=='T2.1':
                cp['critical']=True
        for row in self.result['checkpoints']:row['units']=[]
        s=self.scored()
        self.assertIsNone(s['observed_score'])
        self.assertFalse(s['score_final'])
        self.assertEqual(s['verdict'],'hold')

    def test_entire_metric_excluded_reweights_remaining(self):
        for cp in self.contract['checkpoints']:
            if cp['id'].startswith('S5.'):
                cp.update(units=[],excluded_reason='no exercise in scope')
        for row in self.result['checkpoints']:
            if row['id'].startswith('S5.'):row['units']=[]
        s=self.scored()
        self.assertEqual(s['lower_bound'],100)
        self.assertEqual(s['metrics'][0]['weight'],25)

    def test_cli_outputs_json_and_rejects_template(self):
        c=self.root/'contract.json'; r=self.root/'result.json'
        c.write_text(json.dumps(self.contract));r.write_text(json.dumps(self.result))
        p=subprocess.run([sys.executable,str(SCRIPT),'score',str(c),str(r),'--root',str(self.root)],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)['verdict'],'use')
        self.contract['template_only']=True;c.write_text(json.dumps(self.contract))
        p=subprocess.run([sys.executable,str(SCRIPT),'validate-contract',str(c),'--root',str(self.root)],capture_output=True,text=True)
        self.assertEqual(p.returncode,2)
        self.assertIn('template',p.stderr)


if __name__=='__main__':
    unittest.main()
