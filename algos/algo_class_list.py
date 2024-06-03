def get_algo(algo_name):
    if algo_name.lower()=="fastflow2d".lower():
        from algos.fastflow2d.algo_class import WrapperFastFlow2d
        return WrapperFastFlow2d
    if algo_name.lower()=="patchcore".lower():
        from algos.patchcore.algo_class import WrapperPatchCore
        return WrapperPatchCore
    if algo_name.lower()=="PFM".lower():
        from algos.pefm.algo_class import WrapperPFM
        return WrapperPFM
    if algo_name.lower()=="Fastflow2d_AltUB".lower():
        from algos.altub.algo_class import WrapperFastFlowAltUB
        return WrapperFastFlowAltUB
    if algo_name.lower()=="PEFM".lower():
        from algos.pefm.algo_class import WrapperPEFM
        return WrapperPEFM
    if algo_name.lower()=="CFlow".lower():
        from algos.cflow.algo_class import WrapperCFLOW
        return WrapperCFLOW
    if algo_name.lower()=="MSPBA".lower():
        from algos.mspba.algo_class import WrapperMSPBA
        return WrapperMSPBA
    if algo_name.lower()=="MemSeg".lower():
        from algos.memseg.algo_class import WrapperMemSeg
        return WrapperMemSeg
    if algo_name.lower()=="CDO".lower():
        from algos.cdo.algo_class import WrapperCDO
        return WrapperCDO
    if algo_name.lower()=="CFA".lower():
        from algos.cfa.algo_class import WrapperCFA
        return WrapperCFA
    if algo_name.lower()=="Reverse_Distillation".lower():
        from algos.reverse_distillation.algo_class import WrapperReverseDistillation
        return WrapperReverseDistillation
    if algo_name.lower()=="AST".lower():
        from algos.ast.algo_class import WrapperAST
        return WrapperAST
    if algo_name.lower()=="msflow".lower():
        from algos.msflow.algo_class import WrapperMSFlow
        return WrapperMSFlow
    if algo_name.lower()=="simplenet".lower():
        from algos.simplenet.algo_class import WrapperSimpleNet
        return WrapperSimpleNet
    if algo_name.lower()=="efficientad".lower():
        from algos.efficientad.algo_class import WrapperEfficientAD
        return WrapperEfficientAD
    if algo_name.lower()=="ppdm".lower():
        from algos.ppdm.algo_class import WrapperPPDM
        return WrapperPPDM
    if algo_name.lower()=="ddad".lower():
        from algos.ddad.algo_class import WrapperDDAD
        return WrapperDDAD
    if algo_name.lower()=="random".lower():
        from algos.random.algo_class import WrapperRandom
        return WrapperRandom
    else:
        raise NotImplementedError("Algorithm not found")
