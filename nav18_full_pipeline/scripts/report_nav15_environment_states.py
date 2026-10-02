"""Package the predefined Nav1.5 environment/state matrix; never change scoring."""
from pathlib import Path
import csv
import shutil
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from rdkit import Chem
from scgd.runtime import read_json, write_json, sha256, code_hashes
from diagnose_nav15_environment_states import lipid_distances, verify_frozen

ROOT = BASE/'outputs/nav15_environment_states_20261002'
OLD = BASE/'outputs/nav15_multireference_20261002'
DEST = BASE/'results/nav15_environment_states_20261002'
LABELS = {
    'flecainide_neutral_protein': '氟卡尼中性／纯蛋白（旧基线）',
    'flecainide_N19_plus1_protein': '氟卡尼 N19 +1／纯蛋白',
    'ranolazine_neutral_protein': '雷诺嗪中性／纯蛋白（旧基线）',
    'ranolazine_neutral_lipid': '雷诺嗪中性／脂质占位',
    'ranolazine_N13_plus1_protein': '雷诺嗪 N13 +1／纯蛋白',
    'ranolazine_N13_plus1_lipid': '雷诺嗪 N13 +1／脂质占位',
    'ranolazine_N16_plus1_protein': '雷诺嗪 N16 +1／纯蛋白',
    'ranolazine_N16_plus1_lipid': '雷诺嗪 N16 +1／脂质占位',
}


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def export_examples(name, source, rows):
    dest = DEST/'conditions'/name
    dest.mkdir(parents=True, exist_ok=True)
    for filename in ('report.json', 'results.json', 'manifest.json'):
        shutil.copy2(source/filename, dest/filename)
    winner = min(rows, key=lambda r: (r['affinity_kcal_mol'], r['conformer_id'], r['vina_seed']))
    closest = min(((m['rmsd_A'], r, m['rank']) for r in rows for m in r['modes']),
                  key=lambda x: (x[0], x[1]['conformer_id'], x[1]['vina_seed'], x[2]))
    examples = []
    for label, row, rank in [('original_score_winner', winner, 1), ('closest_reference_diagnostic', closest[1], closest[2])]:
        path = Path(row['pose']).with_suffix('.sdf')
        mol = list(Chem.SDMolSupplier(str(path), removeHs=True))[rank-1]
        with Chem.SDWriter(str(dest/(label+'.sdf'))) as writer:
            writer.write(mol)
        Chem.MolToPDBFile(mol, str(dest/(label+'.pdb')))
        examples.append({'label': label, 'conformer': row['conformer_id'], 'seed': row['vina_seed'],
                         'mode': row['modes'][rank-1], 'source_sha256': sha256(path),
                         'reference_rmsd_selected_for_illustration_only': label.startswith('closest')})
    write_json(dest/'examples.json', examples)


def main():
    protocol = read_json(ROOT/'matrix_protocol.json')
    if protocol['production_code_hashes'] != code_hashes():
        raise ValueError('Original production code changed')
    data, summary = {}, []
    for name, refname in [('flecainide_neutral_protein','rat_6uz0'), ('ranolazine_neutral_protein','rat_8f6p')]:
        path = OLD/refname/'redocking'
        report = read_json(path/'report.json')
        rows = read_json(path/'results.json')
        if not report['complete'] or report['successful_jobs'] != 15:
            raise ValueError('Incomplete historical control')
        data[name] = rows
        summary.append({'condition': name, 'label': LABELS[name], 'reused_historical': True,
                        'expected': 15, 'successful': 15, 'top1_pass': report['top1_pass_count'],
                        'any_mode_pass': report['any_mode_pass_count'],
                        'median_top1_RMSD_A': report['median_top1_rmsd_A'],
                        'median_best_mode_RMSD_A': report['median_best_of_modes_rmsd_A'],
                        'top1_edge_under_0p5A': sum(r['modes'][0]['actual_grid_face_margin_A'] < .5 for r in rows),
                        'diagnostic_passed': report['reference_diagnostic_passed']})
        export_examples(name, path, rows)
    for name in protocol['conditions']:
        path = ROOT/'runs'/name
        report = read_json(path/'report.json')
        if not report['complete'] or report['successful_jobs'] != report['expected_jobs']:
            raise ValueError('Incomplete condition: '+name)
        m = read_json(path/'manifest.json')
        verify_frozen(m['input_hashes'])
        if m['production_code_hashes'] != code_hashes():
            raise ValueError('Scoring changed in '+name)
        rows = read_json(path/'results.json')
        data[name] = rows
        summary.append({'condition': name, 'label': LABELS[name], 'reused_historical': False,
                        'expected': report['expected_jobs'], 'successful': report['successful_jobs'],
                        'top1_pass': report['top1_pass_count'], 'any_mode_pass': report['any_mode_pass_count'],
                        'median_top1_RMSD_A': report['median_top1_rmsd_A'],
                        'median_best_mode_RMSD_A': report['median_best_of_modes_rmsd_A'],
                        'top1_edge_under_0p5A': report['top1_near_actual_grid_edge_0p5A'],
                        'diagnostic_passed': report['reference_diagnostic_passed']})
        export_examples(name, path, rows)
    summary.sort(key=lambda r: list(LABELS).index(r['condition']))
    csv_write(DEST/'summary.csv', summary)
    env = read_json(ROOT/'ranolazine_lipid/prepare_manifest.json')
    lipid = Chem.SDMolSupplier(env['variants']['zwitterion']['lipid_sdf'], removeHs=True)[0]
    matched = []
    for state in ('ranolazine_neutral', 'ranolazine_N13_plus1', 'ranolazine_N16_plus1'):
        bare = {(r['conformer_id'], r['vina_seed']): r for r in data[state+'_protein']}
        with_lipid = {(r['conformer_id'], r['vina_seed']): r for r in data[state+'_lipid']}
        if set(bare) != set(with_lipid):
            raise ValueError('Not a matched environment comparison')
        for key, first in bare.items():
            second = with_lipid[key]
            m1 = read_json(Path(first['pose']).with_suffix('.job.json'))['signature']
            m2 = read_json(Path(second['pose']).with_suffix('.job.json'))['signature']
            for k in ('ligand_sha256','center','size','exhaustiveness','num_modes','seed','cpu','vina_sha256'):
                if m1[k] != m2[k]:
                    raise ValueError('Matched comparison changed '+k)
            mol = Chem.SDMolSupplier(str(Path(first['pose']).with_suffix('.sdf')), removeHs=True)[0]
            geom = lipid_distances(mol, lipid)
            matched.append({'state': state, 'conformer_id': key[0], 'seed': key[1],
                            'protein_top1_RMSD_A': first['top1_rmsd_A'],
                            'lipid_top1_RMSD_A': second['top1_rmsd_A'],
                            'lipid_minus_protein_top1_RMSD_A': second['top1_rmsd_A']-first['top1_rmsd_A'],
                            'protein_top1_pass': first.get('top1_environment_passed', first['top1_passed']),
                            'lipid_top1_pass': second['top1_environment_passed'],
                            'old_pose_fixed_lipid_min_distance_A': geom['min_lipid_heavy_distance_A'],
                            'old_pose_fixed_lipid_heavy_pairs_below_2A': geom['lipid_heavy_pairs_below_2A']})
    csv_write(DEST/'matched_environment_comparisons.csv', matched)
    pair_summary = []
    for state in ('ranolazine_neutral','ranolazine_N13_plus1','ranolazine_N16_plus1'):
        group = [r for r in matched if r['state']==state]
        pair_summary.append({'state':state,'pairs':len(group),
                             'median_matched_RMSD_change_A':float(np.median([r['lipid_minus_protein_top1_RMSD_A'] for r in group])),
                             'protein_top1_clashes_with_omitted_lipid_count':sum(r['old_pose_fixed_lipid_heavy_pairs_below_2A']>0 for r in group),
                             'rescued_passes':sum(not r['protein_top1_pass'] and r['lipid_top1_pass'] for r in group),
                             'lost_passes':sum(r['protein_top1_pass'] and not r['lipid_top1_pass'] for r in group)})
    passed = [r['condition'] for r in summary if r['diagnostic_passed']]
    overall = {'summary': summary, 'matched_environment_summary': pair_summary,
               'new_jobs_successful':sum(r['successful'] for r in summary if not r['reused_historical']),
               'historical_jobs_reused':30, 'diagnostic_passed_conditions':passed,
               'production_code_unchanged':True,'production_gate_released':False,'selectivity_validated':False,
               'recommendation':'Suspend current Nav1.5 docking quantitative ranking and delta reward; retain Nav1.5 functional risk counter-screen.'
                                if not passed else
                                'Preserve passing rat reference condition as limited sensitivity evidence; human/candidate selectivity remains unvalidated.'}
    write_json(DEST/'report.json', overall)
    auditdest = DEST/'preparation_audits'
    auditdest.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'matrix_protocol.json', DEST/'matrix_protocol.json')
    shutil.copy2(ROOT/'ranolazine_lipid/prepare_manifest.json', auditdest/'lipid_manifest.json')
    for state in (ROOT/'states').iterdir():
        if not (state/'prepare_manifest.json').exists():
            continue
        d=auditdest/state.name; d.mkdir(exist_ok=True)
        for name in ('prepare_manifest.json','chemical_audit.json','reference.sdf','sampling_integrity.json'):
            shutil.copy2(state/name,d/name)
        shutil.copy2(state/'sampling/report.json',d/'sampling_report.json')
    shutil.copy2(ROOT/'states/literature_evidence.json', auditdest/'state_literature_evidence.json')
    for extra in ('verification_tests.json','resume_verification.json'):
        if (ROOT/extra).exists():shutil.copy2(ROOT/extra,DEST/extra)
    table='\n'.join('| '+r['label']+f" | {r['successful']}/{r['expected']} | {r['top1_pass']}/{r['expected']} | {r['any_mode_pass']}/{r['expected']} | {r['median_top1_RMSD_A']:.3f} | {r['median_best_mode_RMSD_A']:.3f} | {r['top1_edge_under_0p5A']} |" for r in summary)
    statement=('所有预定条件仍未达到至少 80% 第一名合格的项目标准。建议结束这一轮参数补救，暂停当前 Nav1.5 Vina 定量排名和双靶点差值奖励，保留 Nav1.5 的功能反筛目标。'
               if not passed else '部分条件达到参考姿势标准：'+', '.join(passed)+'。这仅支持对应大鼠参考和指定条件，仍不能释放人源候选筛选或推断选择性。')
    text=f"""# Nav1.5 脂质占位与质子化对照（2026-10-02）

{statement}

本轮完成 90 次新增对接，复用上一轮 30 次中性纯蛋白对照，没有重跑中性基线。原 score.py、scgd/、综合权重和生产门槛不变；没有启动 2,944 个候选的全量筛选。

## 固定方案及结果

每组 5 个去重构象 ×3 个 Vina 种子，原评分、24 Å 框、exhaustiveness=16、最多9个输出、每任务2 CPU。三个 +1 状态各尝试128个独立ETKDG/MMFF构象，完全删除实验坐标再生成；脂质/纯蛋白对照复用相同配体输入、框和随机种子。

合格姿势要求受体坐标系内对称校正重原子 RMSD ≤2 Å、4 Å 蛋白接触召回 ≥0.5、无蛋白重原子距离 <2 Å；有脂质时另外要求无脂质硬碰撞。原接触标准不因加入脂质改变。至少80%第一名通过才算该诊断条件通过，任一输出合格不能替代第一名合格。重复搜索并非独立实验样本。

| 条件 | 完成 | 第一名合格 | 任一输出合格 | 第一名 RMSD 中位数 Å | 最佳输出 RMSD 中位数 Å | 第一名距框边缘 <0.5 Å |
|---|---:|---:|---:|---:|---:|---:|
{table}

逐条件报告、完整数值与原评分最低第一名/参考RMSD最近示例在 [conditions](conditions)；后者仅为说明失败来源的示意，不是推荐排名。配对变化见 [matched_environment_comparisons.csv](matched_environment_comparisons.csv)，汇总见 [report.json](report.json)。不同质子化状态的构象池不同，因此“质子化效应”同时包含新的构象分布和Vina原子分型影响，不能把它解释为纯静电效应。

## 结果的适用范围

- 氟卡尼在非酰胺哌啶 N19 加质子；雷诺嗪两个非酰胺哌嗪 N13/N16 分别建立 +1 微态。文献宏观pKa支持检查质子化，但未给出两个雷诺嗪微态的结合态比例。本轮不按最低分选择所谓真实状态，不运行未经支持的 +2。
- 三个 +1 参考保留全部沉积重原子坐标及碳 R 构型。它们是化学态敏感性对照，不是新测定的实验参考，且两个受体均为大鼠工程化构建。
- 8F6P 的 6OU A2214 共49个重原子完整；本轮只使用NH3+/磷酸O−、净电荷0的头基。另一 CCD 中性头基已准备但未进入本轮计算。
- 该脂质 C40=C41 的沉积几何为 E，与 CCD 的 Z 不一致；重原子完全保留，未为使结果更好而修成 cis。这是固定观测坐标的占位诊断，不能称真实天然POPE或完整膜环境。冲突段距配体约11 Å，最近末端原子距配体3.504 Å。蛋白原PDBQT字节全部保留；无动态膜、水或脂质交换。
- 默认Vina没有显式库仑静电项；改变电荷并不意味着已经建立pH依赖的亲和力模型。对接分数不能替代 IC50、功能抑制或药物心脏安全性。

## 是否撤掉 Nav1.5 反筛

应把两件事分开：当前无法充分验证的 Vina 打分可以暂停作为定量决策；Nav1.5 功能风险目标应保留。某个框内未得到结合姿势不能标注为“对Nav1.5安全”。

建议生成与Nav1.8探索继续，先审 A-803467 原种子及少量结构多样代表。若实现独立的 Nav1.8 探索模式，应保留 nav15_risk=unknown、nav15_score=null、delta=null、原combined_score=null，不能填0或暗改原权重。本轮尚未实现该新模式，原双靶点生产门槛仍关闭。

优先候选需在明确状态、用药时长和电压协议下做配对的人源 Nav1.8/Nav1.5 功能测定；条件允许时检查峰值/晚钠电流及使用依赖性。若要建立计算反筛替代，先做功能数据可用性审计，再考虑有外部验证、适用域和不确定度的简单模型；不能把混杂实验IC50或未测值混成可靠标签。更换对接引擎或柔性受体法属于另一个预先定义的基准任务，不应继续无限追加当前搜索参数。

完整替代方案与文献见 [decision_options.md](decision_options.md)。本轮结果不表明候选无Nav1.5风险，也不证明整个Nav1.5结构研究不可行。

## 复现

配置在 outputs/nav15_environment_states_20261002/configs；全部姿势与日志在同目录 runs。每组可用 scripts/diagnose_nav15_environment_states.py --config <配置> --output <原运行目录> --workers 3 --resume 复用；输入、评分和可执行文件哈希不符时拒绝续跑。

本报告由 scripts/report_nav15_environment_states.py 生成。旧结果未覆盖，候选库未修改，报告和准备审计的文件哈希在 package_manifest.json。
"""
    (DEST/'README.md').write_text(text,encoding='utf-8')
    artifacts={str(p.relative_to(DEST)):sha256(p) for p in sorted(DEST.rglob('*'))
               if p.is_file() and p.name!='package_manifest.json'}
    write_json(DEST/'package_manifest.json',{'artifact_sha256':artifacts,'production_code_hashes':code_hashes(),
                                           'report_script_sha256':sha256(__file__)})
    print('Packaged',len(artifacts),'files. Passing diagnostic conditions:',passed)


if __name__=='__main__':
    main()

