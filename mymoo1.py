# ### 问题设置
import torch
from torch import Tensor
from typing import Union
from botorch.test_functions.base import (
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective import is_non_dominated

class TestProblem1(MultiObjectiveTestProblem, ConstrainedBaseTestProblem):
    dim = 2  # 问题的维度为2
    num_objectives = 2  # 目标数量为2
    num_constraints = 2  # 约束数量为2
    _bounds = [(0.0, 0.0), (1.0, 1.0)]  # 定义输入变量的范围
    _con_bounds = [(0.0, 0.0), (1.0, 1.0)]  # 定义约束变量的范围
    _ref_point = [0.0, 0.0]

    def __init__(
        self,
        noise_std: Union[None, float, list[float]] = None,  # 观测噪声的标准差
        negate: bool = False,  # 是否对目标取反
    ) -> None:
        super().__init__(noise_std=noise_std, negate=negate)  # 调用父类的构造函数

    def _rescaled(self, X: Tensor) -> Tensor:
        x1_min = 0.1
        x1_max = 1.0
        x2_min = 0.0
        x2_max = 5.0
        x1 = (x1_max - x1_min) * X[..., 0] + x1_min # 提取第一个输入变量
        x2 = (x2_max - x2_min) * X[..., 1] + x2_min # 提取第二个输入变量
        return x1, x2
    
    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = self._rescaled(X)
        f1 = x1
        f2 = (1 + x2) / x1
        f1_norm = f1
        f2_norm = f2 / 60.0
        return torch.stack([f1_norm, f2_norm], dim=-1) # 将两个目标函数堆叠在一起
    
    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        x1 = X[..., 0] # 提取第一个输入变量
        x2 = X[..., 1] # 提取第二个输入变量
        g1 = x2 + 9 * x1 - 6
        g2 = -x2 + 9 * x1 - 1
        return torch.stack([g1, g2], dim=-1) # 将两个约束函数堆叠在一起


from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
import matplotlib.pyplot as plt
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.hypervolume import Hypervolume

# 设置数据类型和设备
tkwargs = {
    "dtype": torch.double,  # 设置数据类型为双精度浮点数
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # 设置设备为GPU（如果可用）或CPU
}

NOISE_SE = torch.tensor([1e-5, 1e-5], **tkwargs)  # 定义噪声标准差

def generate_initial_data(n=6):
    # 生成训练数据
    # 生成数据的自变量是从[0, 1]中随机抽取的n个值
    # 返回自变量、真实目标值和加噪目标值
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_x = normalize(train_x, problem.bounds)
    train_obj_true = problem(train_x)  # 计算真实目标值
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE  # 添加噪声
    return train_x, train_obj, train_obj_true


def initialize_model(train_x, train_obj):
    # 初始化模型
    train_x = normalize(train_x, problem.bounds)  # 归一化训练数据，将训练数据缩放到[0, 1]范围内
    models = []  # 创建一个空列表，用于存储每个目标的GP模型
    for i in range(train_obj.shape[-1]):  # 遍历每个目标
        train_y = train_obj[..., i : i + 1]  # 提取第i个目标的训练数据，...表示保留之前所有维度，i:i+1为切片
        train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)  # 设置噪声方差，使用预定义的噪声标准差的平方，full_like表示创建一个形状相同的张量，但是全部填充为同样的值
        models.append(
            SingleTaskGP(
                train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)  # 创建单任务GP模型，并使用标准化变换
            )
        )
    model = ModelListGP(*models)  # 创建模型列表，将所有单任务GP模型组合在一起
    mll = SumMarginalLogLikelihood(model.likelihood, model)  # 定义边际对数似然，用于模型训练
    return mll, model  # 返回边际对数似然和模型

# 初始化问题
problem = TestProblem1(negate=False).to(**tkwargs)

# 生成初始数据
train_x, train_obj, train_obj_true = generate_initial_data(n=36)

# 初始化模型
mll, model = initialize_model(train_x, train_obj)

# 定义qNEHVI采集函数
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf

# 设置采集函数
qNEHVI = qLogNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=problem.ref_point,
    X_baseline=train_x,
    prune_baseline=True
)

# 记录超体积变化
bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
pareto_Y = bd.pareto_Y.cpu().numpy()
hv = bd.compute_hypervolume().item()
hv_history = []

# 迭代100轮
print(problem.bounds)
for iteration in range(100):
    # 优化采集函数
    candidate, acq_value = optimize_acqf(
        acq_function=qNEHVI,
        bounds=problem.bounds.T.clone().detach().to(**tkwargs),
        q=8,
        num_restarts=10,
        raw_samples=1024,
        options={"batch_limit": 16, "maxiter": 500},
        sequential=True,
    )

    # 评估候选点
    new_x = candidate.detach()
    new_obj = problem(new_x)

    # 更新训练数据
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])

    # 重新训练模型
    mll, model = initialize_model(train_x, train_obj)

    # 更新采集函数
    qNEHVI = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point,
        X_baseline=train_x,
        prune_baseline=True
    )

    # 计算当前超体积
    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
    pareto_Y = bd.pareto_Y.cpu().numpy()
    hv_value = bd.compute_hypervolume().item()
    hv_history.append(hv_value)
    
    # 打印当前超体积
    print(f"Iteration {iteration + 1}: Hypervolume = {hv_value}")

# 绘制超体积变化图
plt.figure(figsize=(10, 6))
plt.plot(hv_history, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Hypervolume')
plt.title('Hypervolume over Iterations')
plt.grid(True)
plt.savefig("hypervolume.png")

# 绘制以目标函数值为坐标轴的帕累托前沿
plt.figure(figsize=(10, 6))
plt.scatter(pareto_Y[:, 0], pareto_Y[:, 1] * 60.0, c='r', marker='o')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto Front (Objective Space)')
plt.grid(True)
plt.savefig("pareto_objective.png")

# 绘制以自变量为轴的帕累托前沿
non_dominated_mask = is_non_dominated(train_obj)
pareto_X = train_x[non_dominated_mask].cpu().numpy()
plt.figure(figsize=(10, 6))
x1_min = 0.1
x1_max = 1.0
x2_min = 0.0
x2_max = 5.0
plt.scatter((x1_max - x1_min) * pareto_X[:, 0] + x1_min, (x2_max - x2_min) * pareto_X[:, 1] + x2_min, c='b', marker='x')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.title('Pareto Front (Variable Space)')
plt.grid(True)
plt.savefig("pareto_variable.png")