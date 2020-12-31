import settings

import finalPreprocessing
import Fea_extrac1_4
import Fea_extrac5_9
import Fea_extracA_H
import Fea_extracI_P
import Fea_extracQ_Z

settings.init()


Fea_extrac1_4.isl1()
Fea_extrac5_9.isl2()
Fea_extracA_H.isl3()
Fea_extracI_P.isl4()
Fea_extracQ_Z.isl5()

finalPreprocessing.classify()



