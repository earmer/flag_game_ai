"""
测试文件 - 验证所有模块的基本功能
"""

def test_transformer_model():
    """测试Transformer模型"""
    print("\n" + "="*60)
    print("测试 1: Transformer模型")
    print("="*60)

    try:
        from transformer_model import CTFTransformerConfig, build_ctf_transformer
        import torch

        # 创建模型
        config = CTFTransformerConfig(
            d_model=64,
            nhead=4,
            num_layers=2
        )
        model = build_ctf_transformer(config)

        # 测试前向传播
        B, T = 2, 32
        type_ids = torch.randint(0, 6, (B, T))
        features = torch.randn(B, T, 8)
        padding_mask = torch.zeros(B, T, dtype=torch.bool)
        padding_mask[:, 20:] = True
        my_player_indices = (1, 2, 3)

        logits = model(type_ids, features, padding_mask, my_player_indices)

        assert logits.shape == (B, 3, 5), f"Expected shape (2, 3, 5), got {logits.shape}"
        print(f"✓ 模型创建成功，参数量: {model.count_parameters():,}")
        print(f"✓ 前向传播成功，输出形状: {logits.shape}")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

    return True


def test_population():
    """测试种群管理"""
    print("\n" + "="*60)
    print("测试 2: 种群管理")
    print("="*60)

    try:
        from population import Population, PopulationConfig, Individual
        from transformer_model import CTFTransformerConfig

        # 创建种群
        pop_config = PopulationConfig(population_size=4, elite_size=1)
        population = Population(pop_config)

        # 初始化
        model_config = CTFTransformerConfig(d_model=32, num_layers=1)
        population.initialize_random(model_config)

        assert len(population.individuals) == 4
        print(f"✓ 种群初始化成功，大小: {len(population.individuals)}")

        # 测试统计
        population.individuals[0].fitness = 100.0
        population.individuals[1].fitness = 80.0
        population.individuals[2].fitness = 60.0
        population.individuals[3].fitness = 40.0

        population.sort_by_fitness()
        stats = population.get_statistics()

        print(f"✓ 适应度统计: 平均={stats['fitness']['mean']:.1f}, 最大={stats['fitness']['max']:.1f}")

        # 测试精英选择
        elites = population.get_elites()
        assert len(elites) == 1
        assert elites[0].fitness == 100.0
        print(f"✓ 精英选择成功，精英适应度: {elites[0].fitness}")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

    return True


def test_genetic_ops():
    """测试遗传算子"""
    print("\n" + "="*60)
    print("测试 3: 遗传算子")
    print("="*60)

    try:
        from genetic_ops import (
            AnnealingScheduler, AnnealingConfig,
            tournament_selection, crossover_average, mutate_gaussian
        )
        from population import Population, PopulationConfig, Individual
        from transformer_model import CTFTransformerConfig, build_ctf_transformer

        # 测试退火调度
        annealing_config = AnnealingConfig(initial_temperature=1.0, cooling_rate=0.95)
        scheduler = AnnealingScheduler(annealing_config)

        temp_0 = scheduler.get_temperature(0)
        temp_10 = scheduler.get_temperature(10)

        assert temp_0 == 1.0
        assert temp_10 < temp_0
        print(f"✓ 退火调度: Gen0={temp_0:.3f}, Gen10={temp_10:.3f}")

        # 创建测试种群
        pop_config = PopulationConfig(population_size=4)
        population = Population(pop_config)
        model_config = CTFTransformerConfig(d_model=32, num_layers=1)
        population.initialize_random(model_config)

        # 设置适应度
        for i, ind in enumerate(population.individuals):
            ind.fitness = 100.0 - i * 10

        # 测试选择
        selected = tournament_selection(population.individuals, tournament_size=2, temperature=1.0)
        print(f"✓ 锦标赛选择成功，选中个体适应度: {selected.fitness}")

        # 测试交叉
        parent1 = population.individuals[0]
        parent2 = population.individuals[1]
        child = crossover_average(parent1, parent2, alpha=0.5, new_id="test_child", generation=1)
        print(f"✓ 交叉成功，子代ID: {child.id}")

        # 测试变异
        mutate_gaussian(child, temperature=1.0, mutation_rate=0.1)
        print(f"✓ 变异成功")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

    return True


def test_reward_system():
    """测试奖励系统"""
    print("\n" + "="*60)
    print("测试 4: 奖励系统")
    print("="*60)

    try:
        from reward_system import (
            GameStateSnapshot, AdaptiveRewardSystem,
            SparseRewardCalculator, DenseRewardCalculator, CurriculumScheduler
        )

        # 测试课程学习调度
        curriculum = CurriculumScheduler()
        dense_0, sparse_0 = curriculum.get_weights(0)
        dense_25, sparse_25 = curriculum.get_weights(25)
        dense_50, sparse_50 = curriculum.get_weights(50)

        print(f"✓ 课程学习调度:")
        print(f"  Gen 0:  Dense={dense_0:.1%}, Sparse={sparse_0:.1%}")
        print(f"  Gen 25: Dense={dense_25:.1%}, Sparse={sparse_25:.1%}")
        print(f"  Gen 50: Dense={dense_50:.1%}, Sparse={sparse_50:.1%}")

        # 创建测试状态
        prev_state = GameStateSnapshot(
            timestamp=0.0,
            my_players=[
                {'posX': 5, 'posY': 5, 'hasFlag': False, 'inPrison': False}
            ],
            opp_players=[
                {'posX': 15, 'posY': 15, 'hasFlag': False, 'inPrison': False}
            ],
            my_flags=[],
            opp_flags=[{'posX': 18, 'posY': 18, 'canPickup': True}],
            my_score=0,
            opp_score=0,
            game_over=False,
            winner=None
        )

        current_state = GameStateSnapshot(
            timestamp=1.0,
            my_players=[
                {'posX': 6, 'posY': 6, 'hasFlag': False, 'inPrison': False}
            ],
            opp_players=[
                {'posX': 15, 'posY': 15, 'hasFlag': False, 'inPrison': False}
            ],
            my_flags=[],
            opp_flags=[{'posX': 18, 'posY': 18, 'canPickup': True}],
            my_score=0,
            opp_score=0,
            game_over=False,
            winner=None
        )

        # 测试稀疏奖励
        sparse_calc = SparseRewardCalculator()
        sparse_reward, sparse_breakdown = sparse_calc.calculate(current_state, prev_state)
        print(f"✓ 稀疏奖励计算成功: {sparse_reward:.2f}")

        # 测试密集奖励
        dense_calc = DenseRewardCalculator()
        dense_reward, dense_breakdown, player_breakdowns = dense_calc.calculate(
            current_state, prev_state, my_target_pos=(2, 2)
        )
        print(f"✓ 密集奖励计算成功: {dense_reward:.2f}")

        # 测试自适应奖励系统
        reward_system = AdaptiveRewardSystem()
        reward_system.reset_for_generation(5)
        reward_info = reward_system.calculate_reward(
            current_state, prev_state, my_target_pos=(2, 2)
        )
        print(f"✓ 自适应奖励系统: 总奖励={reward_info.total:.2f}")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

    return True


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始测试所有模块")
    print("="*60)

    results = []
    results.append(("Transformer模型", test_transformer_model()))
    results.append(("种群管理", test_population()))
    results.append(("遗传算子", test_genetic_ops()))
    results.append(("奖励系统", test_reward_system()))

    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + "="*60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查错误信息")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
