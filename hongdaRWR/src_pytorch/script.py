from os.path import join
from findFine import draw_all_plots, init_cross_validate,supplementary_list,train_each_cross_validate,combine_all_cross_validations,draw_top_k,p_value,find_param_cross_validate


if __name__=="__main__":
    from argparse import ArgumentParser
    parser=ArgumentParser(
        description="药物重定位"
    )
    parser.add_argument(
        "actions",
        help="D代表重新划分交叉验证；T代表进行训练和测试；C代表对交叉验证结果进行分析；V代表将验证结果进行可视化,P综合交叉实验，产生预测结果（excel）",
        type=str,
    )
    parser.add_argument(
        "-e","--epochs",
        default=30,
        type=int,
        help="训练回合数目"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="控制随机游走步数"
    )
    parser.add_argument(
        "-a","--alpha",
        default=1.0,
        type=float,
        help="随机游走概率"
    )
    parser.add_argument(
        "-f","--folds",
        default=5,
        type=int,
        help="f倍交叉验证"
    )
    parser.add_argument(
        "--runs",
        default="./runs",
        type=str,
        help="运行结果缓存目录"
    )
    parser.add_argument(
        "--hidden_len",
        type=int,
        default=80,
        help="压缩后的特征维度"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="有效的药物正样本阈值"
    )
    parser.add_argument(
        "-lr",
        type=float,
        default=0.001,
        help="学习率"
    )

    args=parser.parse_args() 
    if "D" in args.actions:
        init_cross_validate(args.folds,dir=args.runs)
    if "F" in args.actions:
        find_param_cross_validate(
            folds=args.folds,
            dir=args.runs,
            epochs=args.epochs,
            lr=args.lr,
            steps=args.step,
            alpha=args.alpha,
            hidden=args.hidden_len
        )
    if "T" in args.actions:
        train_each_cross_validate(
            folds=args.folds,
            dir=args.runs,
            epochs=args.epochs,
            lr=args.lr,
            steps=args.step,
            alpha=args.alpha,
            hidden=args.hidden_len
        )
    if "C" in args.actions:
        combine_all_cross_validations(
            args.folds,
            threshold=args.threshold,
            dir=args.runs
        )
    
    if "V" in args.actions:
        draw_all_plots(args.runs,average_by_drugs=True,save_fig= True,threshold=args.threshold)
        draw_top_k(join(args.runs,"recall.txt"),fig_name=join(args.runs,"top_k.pdf"))
        p_value()

    if "P" in args.actions:
        supplementary_list(args.runs,topk=15,threshold=1)
        supplementary_list(args.runs,"CaseStudy.xlsx",threshold=15,best_first=True)

    
