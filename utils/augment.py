import random
from rdkit import Chem
from rdkit.Chem import AllChem

class MolAugment:
    """针对小分子 SMILES 的增强"""
    def __init__(self, randomize=True):
        self.randomize = randomize

    def randomize_smiles(self, smi):
        """生成随机 SMILES（保持分子结构不变）"""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        if self.randomize:
            return Chem.MolToSmiles(mol, doRandom=True)
        return Chem.MolToSmiles(mol)

    def augment_batch(self, smiles_list, prob=0.5):
        """以概率对 batch 中的 SMILES 做增强"""
        out = []
        for smi in smiles_list:
            if random.random() < prob:
                out.append(self.randomize_smiles(smi))
            else:
                out.append(smi)
        return out


class ProteinAugment:
    """针对蛋白序列的轻度增强"""
    def __init__(self, substitution_prob=0.05):
        # 可以替换氨基酸的保守集合，避免破坏功能
        self.aa = "ACDEFGHIKLMNPQRSTVWY"
        self.sub_prob = substitution_prob

    def random_substitution(self, seq):
        seq = list(seq)
        for i in range(len(seq)):
            if random.random() < self.sub_prob:
                seq[i] = random.choice(self.aa)
        return "".join(seq)

    def augment_batch(self, seq_list, prob=0.5):
        out = []
        for s in seq_list:
            if random.random() < prob:
                out.append(self.random_substitution(s))
            else:
                out.append(s)
        return out
