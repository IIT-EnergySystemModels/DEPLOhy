# =========================
# Developed by
# =========================

#    Andres Ramos
#    Juan F Gutierrez-Guerra
#    Instituto de Investigacion Tecnologica
#    Escuela Tecnica Superior de Ingenieria - ICAI
#    UNIVERSIDAD PONTIFICIA COMILLAS
#    Alberto Aguilera 23
#    28015 Madrid, Spain
#    Andres.Ramos@comillas.edu
#    jgutierrez@comillas.edu

# Libraries
import datetime
import os
import math
import time
import psutil
import platform
import socket
import pandas               as     pd
import numpy                as     np
import matplotlib.pyplot    as     plt
import pyomo.environ        as     pyo
import re                   as     regex
from   pyomo.environ        import Set, Param, Var, Binary, UnitInterval, NonNegativeIntegers, PositiveIntegers, NonNegativeReals, Reals, Any, Constraint, ConcreteModel, Objective, minimize
from   pyomo.opt            import SolverFactory
from   pyomo.dataportal     import DataPortal
from   collections          import defaultdict

for i in range(0, 124):
    print('-', end="")

print('\nDEPLOhy - Determining electrolyzer long-term planning and operation for hydrogen expansion - Version 6.3 - July 6, 2026')
print('#### Non-commercial use only ####')

for i in range(0, 124):
    print('-', end="")

StartTime  = time.time()
DirName    = os.path.dirname(__file__)
CaseName   = 'Expansion'
SolverName = 'gurobi'
_path      = os.path.join(DirName, CaseName)

# =========================
# Model declaration
# =========================
mDEPLOhy = ConcreteModel('DEPLOhy - Determining electrolyzer long-term planning and operation for hydrogen expansion - Version 6.3 - July 6, 2026')

# Reading the sets
dictSets = DataPortal()
dictSets.load(filename=_path+'/DEPLOhy_Dict_Period_'            +CaseName+'.csv', set='p'   , format='set')
dictSets.load(filename=_path+'/DEPLOhy_Dict_Scenario_'          +CaseName+'.csv', set='sc'  , format='set')
dictSets.load(filename=_path+'/DEPLOhy_Dict_LoadLevel_'         +CaseName+'.csv', set='n'   , format='set')
dictSets.load(filename=_path+'/DEPLOhy_Dict_Component_'         +CaseName+'.csv', set='g'   , format='set')
dictSets.load(filename=_path+'/DEPLOhy_Dict_Technology_'        +CaseName+'.csv', set='gt'  , format='set')
dictSets.load(filename=_path+'/DEPLOhy_Dict_Node_'              +CaseName+'.csv', set='nd'  , format='set')
dictSets.load(filename=_path+'/DEPLOhy_Dict_Circuit_'           +CaseName+'.csv', set='cc'  , format='set')
dictSets.load(filename=_path+'/DEPLOhy_Dict_LoadRange_'         +CaseName+'.csv', set='lr'  , format='set')
dictSets.load(filename=_path+'/DEPLOhy_Dict_LifeStage_'         +CaseName+'.csv', set='ls'  , format='set')
dictSets.load(filename=_path+'/DEPLOhy_Dict_PeriodToLifeStage_' +CaseName+'.csv', set='p2l' , format='set')
dictSets.load(filename=_path+'/DEPLOhy_Dict_ChargeLevel_'       +CaseName+'.csv', set='cl'  , format='set')
dictSets.load(filename=_path+'/DEPLOhy_Dict_MarketPeriod_'      +CaseName+'.csv', set='m'   , format='set')

mDEPLOhy.pp  = Set(initialize=dictSets['p'   ], ordered=True,  doc='periods'      )
mDEPLOhy.scc = Set(initialize=dictSets['sc'  ], ordered=True,  doc='scenarios'    )
mDEPLOhy.nn  = Set(initialize=dictSets['n'   ], ordered=True,  doc='load levels'  )
mDEPLOhy.gg  = Set(initialize=dictSets['g'   ], ordered=True,  doc='components'   )
mDEPLOhy.gt  = Set(initialize=dictSets['gt'  ], ordered=False, doc='technologies' )
mDEPLOhy.nd  = Set(initialize=dictSets['nd'  ], ordered=False, doc='nodes'        )
mDEPLOhy.ni  = Set(initialize=dictSets['nd'  ], ordered=False, doc='nodes'        )
mDEPLOhy.nf  = Set(initialize=dictSets['nd'  ], ordered=False, doc='nodes'        )
mDEPLOhy.cc  = Set(initialize=dictSets['cc'  ], ordered=True,  doc='circuits'     )
mDEPLOhy.lr  = Set(initialize=dictSets['lr'  ], ordered=True,  doc='load range'   )
mDEPLOhy.ls  = Set(initialize=dictSets['ls'  ], ordered=True,  doc='life stage'   )
mDEPLOhy.p2l = Set(initialize=dictSets['p2l' ], ordered=True,  doc='period2ls'    )
mDEPLOhy.cl  = Set(initialize=dictSets['cl'  ], ordered=True,  doc='charge level' )
mDEPLOhy.mm  = Set(initialize=dictSets['m'   ], ordered=True,  doc='market period')

# Reading data from CSV files
dfOption                = pd.read_csv(_path+'/DEPLOhy_Data_Option_'                    +CaseName+'.csv', index_col=[0    ])
dfParameter             = pd.read_csv(_path+'/DEPLOhy_Data_Parameter_'                 +CaseName+'.csv', index_col=[0    ])
dfDuration              = pd.read_csv(_path+'/DEPLOhy_Data_Duration_'                  +CaseName+'.csv', index_col=[0    ])
dfPeriod                = pd.read_csv(_path+'/DEPLOhy_Data_Period_'                    +CaseName+'.csv', index_col=[0    ])
dfScenario              = pd.read_csv(_path+'/DEPLOhy_Data_Scenario_'                  +CaseName+'.csv', index_col=[0,1  ])
dfSpotPrice             = pd.read_csv(_path+'/DEPLOhy_Data_SpotPrice_'                 +CaseName+'.csv', index_col=[0,1,2])
dfReservesPrice         = pd.read_csv(_path+'/DEPLOhy_Data_ReservesPrice_'             +CaseName+'.csv', index_col=[0,1,2])
dfEnergyActivation      = pd.read_csv(_path+'/DEPLOhy_Data_EnergyActivation_'          +CaseName+'.csv', index_col=[0,1,2])
dfECI                   = pd.read_csv(_path+'/DEPLOhy_Data_ElectricityCarbonIntensity_'+CaseName+'.csv', index_col=[0,1,2])
dfComponent             = pd.read_csv(_path+'/DEPLOhy_Data_Component_'                 +CaseName+'.csv', index_col=[0    ])
dfCurve                 = pd.read_csv(_path+'/DEPLOhy_Data_ElectrolyzerCurve_'         +CaseName+'.csv', index_col=[0,1,2])
dfCompression           = pd.read_csv(_path+'/DEPLOhy_Data_Compression_'               +CaseName+'.csv', index_col=[0    ])
dfVariableMaxPower      = pd.read_csv(_path+'/DEPLOhy_Data_VariableMaxGeneration_'     +CaseName+'.csv', index_col=[0,1,2])
dfVariableMaxCharge     = pd.read_csv(_path+'/DEPLOhy_Data_VariableMaxConsumption_'    +CaseName+'.csv', index_col=[0,1,2])
dfVariableMinStorage    = pd.read_csv(_path+'/DEPLOhy_Data_variableMinStorage_'        +CaseName+'.csv', index_col=[0,1,2])
dfVariableMaxStorage    = pd.read_csv(_path+'/DEPLOhy_Data_variableMaxStorage_'        +CaseName+'.csv', index_col=[0,1,2])
dfVariableMinH2Storage  = pd.read_csv(_path+'/DEPLOhy_Data_variableMinH2Storage_'      +CaseName+'.csv', index_col=[0,1,2])
dfVariableMaxH2Storage  = pd.read_csv(_path+'/DEPLOhy_Data_variableMaxH2Storage_'      +CaseName+'.csv', index_col=[0,1,2])
dfVariableMinOutflows   = pd.read_csv(_path+'/DEPLOhy_Data_VariableMinOutflows_'       +CaseName+'.csv', index_col=[0,1,2])
dfVariableMaxOutflows   = pd.read_csv(_path+'/DEPLOhy_Data_VariableMaxOutflows_'       +CaseName+'.csv', index_col=[0,1,2])
dfNetwork               = pd.read_csv(_path+'/DEPLOhy_Data_Network_'                   +CaseName+'.csv', index_col=[0,1,2])
dfMarketPeriod          = pd.read_csv(_path+'/DEPLOhy_Data_MarketPeriod_'              +CaseName+'.csv', index_col=[0,1  ])
dfGridTariff            = pd.read_csv(_path+'/DEPLOhy_Data_GridTariff_'                +CaseName+'.csv', index_col=[0,1  ])

# Substitute NaN by 0
dfOption.fillna               (0.0, inplace=True)
dfParameter.fillna            (0.0, inplace=True)
dfDuration.fillna             (0.0, inplace=True)
dfPeriod.fillna               (0.0, inplace=True)
dfScenario.fillna             (0.0, inplace=True)
dfSpotPrice.fillna            (0.0, inplace=True)
dfReservesPrice.fillna        (0.0, inplace=True)
dfEnergyActivation.fillna     (0.0, inplace=True)
dfECI.fillna                  (0.0, inplace=True)
dfComponent.fillna            (0.0, inplace=True)
dfCurve.fillna                (0.0, inplace=True)
dfCompression.fillna          (0.0, inplace=True)
dfVariableMaxPower.fillna     (0.0, inplace=True)
dfVariableMaxCharge.fillna    (0.0, inplace=True)
dfVariableMinStorage.fillna   (0.0, inplace=True)
dfVariableMaxStorage.fillna   (0.0, inplace=True)
dfVariableMinH2Storage.fillna (0.0, inplace=True)
dfVariableMaxH2Storage.fillna (0.0, inplace=True)
dfVariableMinOutflows.fillna  (0.0, inplace=True)
dfVariableMaxOutflows.fillna  (0.0, inplace=True)
dfNetwork.fillna              (0.0, inplace=True)
dfMarketPeriod.fillna         (0.0, inplace=True)
dfGridTariff.fillna           (0.0, inplace=True)

dfVariableMaxPower     = dfVariableMaxPower.where    (dfVariableMaxPower     > 0.0, 0.0)
dfVariableMaxCharge    = dfVariableMaxCharge.where   (dfVariableMaxCharge    > 0.0, 0.0)
dfVariableMinStorage   = dfVariableMinStorage.where  (dfVariableMinStorage   > 0.0, 0.0)
dfVariableMaxStorage   = dfVariableMaxStorage.where  (dfVariableMaxStorage   > 0.0, 0.0)
dfVariableMinH2Storage = dfVariableMinH2Storage.where(dfVariableMinH2Storage > 0.0, 0.0)
dfVariableMaxH2Storage = dfVariableMaxH2Storage.where(dfVariableMaxH2Storage > 0.0, 0.0)
dfVariableMinOutflows  = dfVariableMinOutflows.where (dfVariableMinOutflows  > 0.0, 0.0)
dfVariableMaxOutflows  = dfVariableMaxOutflows.where (dfVariableMaxOutflows  > 0.0, 0.0)

# General parameters
pIndBinInvest         = dfOption               ['IndBinInvestment'   ].iloc[0].astype('int')         # Indicator of binary investment decisions,          0 continuous    - 1 binary - 2 neglect investment
pIndBinRetire         = dfOption               ['IndBinRetirement'   ].iloc[0].astype('int')         # Indicator of binary retirement decisions,          0 continuous    - 1 binary - 2 neglect retirement
pIndBinOperation      = dfOption               ['IndBinOperation'    ].iloc[0].astype('int')         # Indicator of binary operation  decisions,          0 continuous    - 1 binary
pIndReserves          = dfOption               ['IndReserves'        ].iloc[0].astype('int')         # Indicator of reserves provision mode,              0 no reserves   - 1 with reserves
pIndDegradation       = dfOption               ['IndDegradation'     ].iloc[0].astype('int')         # Indicator of degradation losses,                   0 lossless      - 1 degradation losses
pIndPiecewise         = dfOption               ['IndPiecewise'       ].iloc[0].astype('int')         # Indicator of EZ curve piecewise linear fit,        0 linear fit    - 1 piecewise linear fit
pIndTwoStates         = dfOption               ['IndTwoStates'       ].iloc[0].astype('int')         # Indicator of two-state EZ model (On-Standby),      0 three-state   - 1 two-state
pIndLCF               = dfOption               ['IndLCF'             ].iloc[0].astype('int')         # Indicator of compliance w/ low carbon fuel certif, 0 unconstrained - 1 constrained
pIndRFNBO             = dfOption               ['IndRFNBO'           ].iloc[0].astype('int')         # Indicator of compliance w/ RFNBO certification,    0 unconstrained - 1 constrained
pIndCapFactor         = dfOption               ['IndCapFactor'       ].iloc[0].astype('int')         # Indicator of capacity factor constraint,           0 unconstrained - 1 constrained
pIndFigures           = dfOption               ['IndFigures'         ].iloc[0].astype('int')         # Indicator of figures generation,                   0 no figures    - 1 generate figures
pFossilComparator     = dfParameter            ['FossilComparator'   ].iloc[0]                       # fossil fuel comparator                   [gCO2eq/MJ]
pSavingTarget         = dfParameter            ['GHESavingTarget'    ].iloc[0]                       # GHE savings target                       [p.u.]
pTimeStep             = dfParameter            ['TimeStep'           ].iloc[0].astype('int')         # duration of the unit time step           [h]
pAnnualDiscRate       = dfParameter            ['AnnualDiscountRate' ].iloc[0]                       # annual discount rate                     [p.u.]
pEconomicBaseYear     = dfParameter            ['EconomicBaseYear'   ].iloc[0]                       # economic base year                       [year]
pAnnualOutflows       = dfParameter            ['AnnualProduction'   ].iloc[0]                       # fixed period H2 production               [tonH2/period]
pMinCapFactor         = dfParameter            ['MinCapFactor'       ].iloc[0]                       # min EZ capacity factor                   [p.u.]
pGCPCapacity          = dfParameter            ['GCPCapacity'        ].iloc[0] * 1e-3                # grid connection point capacity           [GW]
pVAT                  = dfParameter            ['VAT'                ].iloc[0]                       # value added tax                          [p.u.]
pEnergyTax            = dfParameter            ['ElectricityTax'     ].iloc[0]                       # other electricity taxes                  [p.u.]
pDuration             = dfDuration             ['Duration'           ] * pTimeStep                   # duration of load levels                  [h]
pPeriodWeight         = dfPeriod               ['Weight'             ].astype('int')                 # weights of periods                       [p.u.]
pMinAnnualOutflows    = dfPeriod               ['MinAnnualProd'      ]                               # minimum H2 production per period         [tonH2/period]
pAPRE                 = dfPeriod               ['APRE'               ]                               # average proportion of ren electricity    [p.u.]
pNRH                  = dfPeriod               ['NRH'                ]                               # number of renewable electricity hours    [h]
pScenProb             = dfScenario             ['Probability'        ].astype('float')               # probabilities of scenarios               [p.u.]
pSpotPrice            = dfSpotPrice            [mDEPLOhy.nd          ] * 1e-3                        # spot price per node                      [MEUR/GWh]
pResUpPrice           = dfReservesPrice        ['CapacityUp'         ] * 1e-3                        # balancing capacity       upwards price   [MEUR/GW ]
pResDwPrice           = dfReservesPrice        ['CapacityDw'         ] * 1e-3                        # balancing capacity     downwards price   [MEUR/GW ]
pEnergyUpPrice        = dfReservesPrice        ['EnergyUp'           ] * 1e-3                        # balancing energy         upwards price   [MEUR/GWh]
pEnergyDwPrice        = dfReservesPrice        ['EnergyDw'           ] * 1e-3                        # balancing energy       downwards price   [MEUR/GWh]
pActivationUp         = dfEnergyActivation     ['ActivationUp'       ]                               # activation requirement   upwards         [p.u.]
pActivationDw         = dfEnergyActivation     ['ActivationDw'       ]                               # activation requirement downwards         [p.u.]
pMaxRatioDwUp         = dfEnergyActivation     ['MaxRatioDwUp'       ]                               # max ratio between down and up capacity   [p.u.]
pECI                  = dfECI                  [mDEPLOhy.nd          ]                               # electricity carbon intensity per node    [gCO2eq/MJe]
pVarMaxPower          = dfVariableMaxPower     [mDEPLOhy.gg          ]                               # variable maximum power                   [p.u.]
pVarMaxCharge         = dfVariableMaxCharge    [mDEPLOhy.gg          ]                               # variable maximum charge                  [p.u.]
pVarMinStorage        = dfVariableMinStorage   [mDEPLOhy.gg          ]                               # variable minimum electricity storage     [p.u.]
pVarMaxStorage        = dfVariableMaxStorage   [mDEPLOhy.gg          ]                               # variable maximum electricity storage     [p.u.]
pVarMinH2Storage      = dfVariableMinH2Storage [mDEPLOhy.gg          ]                               # variable minimum h2 storage              [p.u.]
pVarMaxH2Storage      = dfVariableMaxH2Storage [mDEPLOhy.gg          ]                               # variable maximum h2 storage              [p.u.]
pVarMinOutflows       = dfVariableMinOutflows  [mDEPLOhy.gg          ]                               # variable product outflows                [p.u.]
pVarMaxOutflows       = dfVariableMaxOutflows  [mDEPLOhy.gg          ]                               # variable product outflows                [p.u.]
pMarketPeriod         = dfMarketPeriod         [mDEPLOhy.mm          ]                               # hourly market period                     [-]
pPowerTariff          = dfGridTariff           ['PowerTerm'          ] * 1e-3                        # power term grid tariff                   [MEUR/GW/year]
pEnergyTerm           = dfGridTariff           ['EnergyTerm'         ] * 1e-3                        # power term grid tariff                   [MEUR/GWh]
pFixedPower           = dfGridTariff           ['FixedPower'         ] * 1e-3                        # fixed contracted power                   [GW]

# Compute as the mean over the time step load levels and assign it to active load levels
pSpotPrice            = pSpotPrice.rolling      (pTimeStep).mean()
pResUpPrice           = pResUpPrice.rolling     (pTimeStep).mean()
pResDwPrice           = pResDwPrice.rolling     (pTimeStep).mean()
pEnergyUpPrice        = pEnergyUpPrice.rolling  (pTimeStep).mean()
pEnergyDwPrice        = pEnergyDwPrice.rolling  (pTimeStep).mean()
pActivationUp         = pActivationUp.rolling   (pTimeStep).mean()
pActivationDw         = pActivationDw.rolling   (pTimeStep).mean()
pMaxRatioDwUp         = pMaxRatioDwUp.rolling   (pTimeStep).mean()
pECI                  = pECI.rolling            (pTimeStep).mean()
pVarMaxPower          = pVarMaxPower.rolling    (pTimeStep).mean()
pVarMaxCharge         = pVarMaxCharge.rolling   (pTimeStep).mean()
pVarMinStorage        = pVarMinStorage.rolling  (pTimeStep).mean()
pVarMaxStorage        = pVarMaxStorage.rolling  (pTimeStep).mean()
pVarMinH2Storage      = pVarMinH2Storage.rolling(pTimeStep).mean()
pVarMaxH2Storage      = pVarMaxH2Storage.rolling(pTimeStep).mean()
pVarMinOutflows       = pVarMinOutflows.rolling (pTimeStep).mean()
pVarMaxOutflows       = pVarMaxOutflows.rolling (pTimeStep).mean()

pSpotPrice.fillna      (0.0, inplace=True)
pResUpPrice.fillna     (0.0, inplace=True)
pResDwPrice.fillna     (0.0, inplace=True)
pEnergyUpPrice.fillna  (0.0, inplace=True)
pEnergyDwPrice.fillna  (0.0, inplace=True)
pActivationUp.fillna   (0.0, inplace=True)
pActivationDw.fillna   (0.0, inplace=True)
pMaxRatioDwUp.fillna   (0.0, inplace=True)
pECI.fillna            (0.0, inplace=True)
pVarMaxPower.fillna    (0.0, inplace=True)
pVarMaxCharge.fillna   (0.0, inplace=True)
pVarMinStorage.fillna  (0.0, inplace=True)
pVarMaxStorage.fillna  (0.0, inplace=True)
pVarMinH2Storage.fillna(0.0, inplace=True)
pVarMaxH2Storage.fillna(0.0, inplace=True)
pVarMinOutflows.fillna (0.0, inplace=True)
pVarMaxOutflows.fillna (0.0, inplace=True)
pMarketPeriod.fillna   (0.0, inplace=True)
pPowerTariff.fillna    (0.0, inplace=True)
pEnergyTerm.fillna     (0.0, inplace=True)
pFixedPower.fillna     (0.0, inplace=True)

if pTimeStep > 1:
    # assign duration 0 to load levels not being considered, active load levels are at the end of every pTimeStep
    for i in range(pTimeStep-2,-1,-1):
        pDuration.iloc[[range(i,len(mDEPLOhy.nn),pTimeStep)]] = 0

# Components' parameters
pUnitToNode         = dfComponent['Node'                ]                                                               # unit location in node
pUnitToTechnology   = dfComponent['Technology'          ]                                                               # unit association to technology
pIndBinUnitInvest   = dfComponent['BinaryInvestment'    ]                                                               # binary unit investment  decision        [Yes]
pIndBinUnitRetire   = dfComponent['BinaryRetirement'    ]                                                               # binary unit retirement  decision        [Yes]
pInitialPeriod      = dfComponent['InitialPeriod'       ]                                                               # initial period                          [year]
pFinalPeriod        = dfComponent['FinalPeriod'         ]                                                               # final   period                          [year]
pRatedMaxPower      = dfComponent['MaxPower'            ] * 1e-3                                                        # rated maximum power                     [GW]
pRatedMaxCharge     = dfComponent['MaxCharge'           ] * 1e-3                                                        # maximum ESS charge                      [GW]
pRatedMinCharge     = dfComponent['MinChargeRatio'      ] * pRatedMaxCharge                                             # minimum ESS charge                      [GW]
pH2RatedMaxOutflow  = dfComponent['H2MaxOutflow'        ] * 1e-3                                                        # maximum H2 outflow                      [tonH2]
pH2RatedMinOutflow  = dfComponent['H2MinOutflow'        ] * 1e-3                                                        # minimum H2 outflow                      [tonH2]
pInitialInventory   = dfComponent['InitialStorage'      ]                                                               # initial ESS storage                     [GWh]
pRatedMaxStorage    = dfComponent['MaxStorage'          ]                                                               # maximum ESS storage                     [GWh]
pRatedMinStorage    = dfComponent['MinStorage'          ]                                                               # minimum ESS storage                     [GWh]
pH2InitialInventory = dfComponent['H2InitialStorage'    ] * 1e-3                                                        # initial H2 storage                      [tonH2]
pH2RatedMaxStorage  = dfComponent['H2MaxStorage'        ] * 1e-3                                                        # maximum H2 storage                      [tonH2]
pH2RatedMinStorage  = dfComponent['H2MinStorage'        ] * 1e-3                                                        # minimum H2 storage                      [tonH2]
pEfficiency         = dfComponent['Efficiency'          ]                                                               # ESS efficiency                          [p.u.]
pPeriodDegRate      = dfComponent['PeriodDegRate'       ]                                                               # period degradation rate                 [p.u.]
pSBConsumption      = dfComponent['SBConsumption'       ] * 1e-3                                                        # BoP consumption during stand-by state   [GW]
pBoPFactor          = dfComponent['BoPFactor'           ]                                                               # ratio between stack and BoP capacities  [p.u.]
pOMCost             = dfComponent['OMFixedCost'         ]                                                               # O&M annual      fixed cost              [MEUR/MW/year]
pStartUpCost        = dfComponent['StartUpCost'         ]                                                               # EZ startup            cost              [kEUR]
pShutDownCost       = dfComponent['ShutDownCost'        ]                                                               # EZ shutdown           cost              [kEUR]
pInvestCost         = dfComponent['FixedInvestmentCost' ] * dfComponent['FixedChargeRate']                              # unit annualized fixed cost              [MEUR/MW]
pRetireCost         = dfComponent['FixedRetirementCost' ] * dfComponent['FixedChargeRate']                              # unit fixed retirement cost              [MEUR/MW]
pLowerInvest        = dfComponent['InvestmentLo'        ]                                                               # lower bound of the investment decision  [p.u.]
pUpperInvest        = dfComponent['InvestmentUp'        ]                                                               # upper bound of the investment decision  [p.u.]
pLowerRetire        = dfComponent['RetirementLo'        ]                                                               # lower bound of the retirement decision  [p.u.]
pUpperRetire        = dfComponent['RetirementUp'        ]                                                               # upper bound of the retirement decision  [p.u.]
pPPAPrice           = dfComponent['PPAprice'            ] * 1e-3                                                        # PPA fixed price                         [MEUR/GWh]
pOutflowsRampUp     = dfComponent['OutflowsRampUp'      ] * 1e-3                                                        # H2 outflows ramp up   rate              [tonH2/h]
pOutflowsRampDw     = dfComponent['OutflowsRampDw'      ] * 1e-3 * -1                                                   # H2 outflows ramp down rate              [tonH2/h]

# Electrolyzer curve parameters
pSE                 = dfCurve    ['Efficiency'          ] * 1e-3                                                        # EZ system partial efficiency            [GWh/ton]

# Compression parameters
pHeatCapacity       = dfCompression['Cp'                ]                                                               # specific heat capacity                  [J/kg.K]
pGamma              = dfCompression['Gamma'             ]                                                               # isentropic coefficient                  [-]
pIsEfficiency       = dfCompression['IsEfficiency'      ]                                                               # compressor isentropic efficiency        [p.u.]
pMechEfficiency     = dfCompression['MechEfficiency'    ]                                                               # compressor mechanical efficiency        [p.u.]
pInletT             = dfCompression['T_in'              ]                                                               # compressor inlet  temperature           [ºC]
pInletP             = dfCompression['P_in'              ]                                                               # compressor inlet  pressure              [bar]
pOutletP            = dfCompression['P_out'             ]                                                               # compressor outlet pressure              [bar]


ReadingDataTime = time.time() - StartTime
StartTime       = time.time()
print('\nReading    input data                          ... ', round(ReadingDataTime), 's')

# Replacing pUpperInvest = 0.0 and pUpperRetire = 0.0 by 1.0
pUpperInvest = pUpperInvest.where(pUpperInvest  > 0.0, other=1.0)
pUpperRetire = pUpperRetire.where(pUpperRetire  > 0.0, other=1.0)

# Getting the branches from the network data
sBr     = [(ni, nf) for (ni, nf, cc) in dfNetwork.index]
# Dropping duplicate keys
sBrList = [(ni, nf) for  n, (ni, nf) in enumerate(sBr) if (ni, nf) not in sBr[:n]]

# Getting the points from the electrolyzer curve data
sCr     = [(lr, ls, gg) for (lr, ls, gg) in dfCurve.index]

# Inverse index node to unit
pNodeToUnit   = pUnitToNode.reset_index().set_index( 'Node').set_axis(['Unit'], axis=1, copy=False)[['Unit']]
pNodeToUnit   = pNodeToUnit.loc[pNodeToUnit[                           'Unit'].isin(mDEPLOhy.gg)]
pNode2Unit    = pNodeToUnit.reset_index().set_index(['Node',           'Unit'])
mDEPLOhy.n2g  = Set(initialize=pNode2Unit.index      , ordered=False, doc='node to unit')

# Subsets
mDEPLOhy.p    = Set(initialize=mDEPLOhy.pp           , ordered=True , doc='periods               ', filter=lambda mDEPLOhy,pp  : pp     in mDEPLOhy.pp            and  pPeriodWeight     [pp] >  0.0                                         )
mDEPLOhy.sc   = Set(initialize=mDEPLOhy.scc          , ordered=True , doc='scenarios             ', filter=lambda mDEPLOhy,scc : scc    in mDEPLOhy.scc                                                                                      )
mDEPLOhy.ps   = Set(dimen=2, initialize=[(p,sc)                                                            for    (p,sc), prob          in pScenProb.items()                                                                  ], ordered=True)
mDEPLOhy.n    = Set(initialize=mDEPLOhy.nn           , ordered=True , doc='load levels           ', filter=lambda mDEPLOhy,nn  : nn     in mDEPLOhy.nn            and  pDuration         [nn] >  0                                           )
mDEPLOhy.n2   = Set(initialize=mDEPLOhy.nn           , ordered=True , doc='load levels           ', filter=lambda mDEPLOhy,nn  : nn     in mDEPLOhy.nn            and  pDuration         [nn] >  0                                           )
mDEPLOhy.m    = Set(initialize=mDEPLOhy.mm           , ordered=True , doc='market periods        ', filter=lambda mDEPLOhy,mm  : mm     in mDEPLOhy.mm                                                                                       )
mDEPLOhy.g    = Set(initialize=mDEPLOhy.gg           , ordered=False, doc='all              units', filter=lambda mDEPLOhy,gg  : gg     in mDEPLOhy.gg            and (pRatedMaxPower    [gg] >  0.0 or  pRatedMaxCharge[gg]     > 0.0  or  pH2RatedMaxStorage[gg]) > 0.0 and pInitialPeriod[gg] <= mDEPLOhy.p.last() and pFinalPeriod[gg] >= mDEPLOhy.p.first() and pUnitToNode.reset_index().set_index(['index']).isin(mDEPLOhy.nd)['Node'][gg]) # excludes units with empty node
mDEPLOhy.gp   = Set(initialize=mDEPLOhy.g            , ordered=False, doc='generating       units', filter=lambda mDEPLOhy,g   : g      in mDEPLOhy.g             and  pRatedMaxPower     [g] >  0.0                                         )
mDEPLOhy.ppa  = Set(initialize=mDEPLOhy.g            , ordered=False, doc='PPA              units', filter=lambda mDEPLOhy,g   : g      in mDEPLOhy.g             and  pPPAPrice          [g] >  0.0                                         )
mDEPLOhy.re   = Set(initialize=mDEPLOhy.g            , ordered=False, doc='RES              units', filter=lambda mDEPLOhy,g   : g      in mDEPLOhy.g             and  pRatedMaxCharge    [g] == 0.0 and pRatedMaxStorage[g]    == 0.0  and pH2RatedMaxStorage[g]  == 0.0 and pRatedMinCharge[g] == 0.0 and pRatedMaxPower[g] > 0.0 and pPPAPrice[g] == 0.0)
mDEPLOhy.st   = Set(initialize=mDEPLOhy.g            , ordered=False, doc='all storage      units', filter=lambda mDEPLOhy,g   : g      in mDEPLOhy.g             and  pRatedMaxCharge    [g] >  0.0 or  pRatedMaxStorage[g]     > 0.0  or  pH2RatedMaxStorage[g]   > 0.0 or  pH2RatedMaxOutflow[g])
mDEPLOhy.es   = Set(initialize=mDEPLOhy.st           , ordered=False, doc='ESS              units', filter=lambda mDEPLOhy,st  : st     in mDEPLOhy.st            and  pH2RatedMaxStorage[st] == 0.0                                         )
mDEPLOhy.hs   = Set(initialize=mDEPLOhy.st           , ordered=False, doc='H2 storage       units', filter=lambda mDEPLOhy,st  : st     in mDEPLOhy.st            and  pH2RatedMaxStorage[st] >  0.0                                         )
mDEPLOhy.el   = Set(initialize=mDEPLOhy.g            , ordered=False, doc='electrolyzer     units', filter=lambda mDEPLOhy,g   : g      in mDEPLOhy.g             and  pRatedMinCharge    [g] >  0.0                                         )
mDEPLOhy.bp   = Set(initialize=mDEPLOhy.g            , ordered=False, doc='balance of plant      ', filter=lambda mDEPLOhy,g   : g      in mDEPLOhy.g             and  pBoPFactor         [g] >  0.0                                         )
mDEPLOhy.gc   = Set(initialize=mDEPLOhy.g            , ordered=False, doc='candidate        units', filter=lambda mDEPLOhy,g   : g      in mDEPLOhy.g             and  pInvestCost        [g] >  0.0                                         )
mDEPLOhy.ec   = Set(initialize=mDEPLOhy.gc           , ordered=False, doc='candidate ESS    units', filter=lambda mDEPLOhy,gc  : gc     in mDEPLOhy.gc            and  pRatedMaxCharge   [gc] != 0.0                                         )
mDEPLOhy.hc   = Set(initialize=mDEPLOhy.gc           , ordered=False, doc='candidate hs     units', filter=lambda mDEPLOhy,gc  : gc     in mDEPLOhy.gc            and  pH2RatedMaxStorage[gc] != 0.0                                         )
mDEPLOhy.rc   = Set(initialize=mDEPLOhy.gc           , ordered=False, doc='candidate not st units', filter=lambda mDEPLOhy,gc  : gc     in mDEPLOhy.gc            and  pRatedMaxCharge   [gc] == 0.0 and pH2RatedMaxStorage[gc] == 0.0       )
mDEPLOhy.gd   = Set(initialize=mDEPLOhy.g            , ordered=False, doc='retirement       units', filter=lambda mDEPLOhy,g   : g      in mDEPLOhy.g             and  pRetireCost        [g] != 0.0                                         )
mDEPLOhy.ed   = Set(initialize=mDEPLOhy.gd           , ordered=False, doc='retirement ess   units', filter=lambda mDEPLOhy,gd  : gd     in mDEPLOhy.gd            and  pRatedMaxCharge   [gd] != 0.0                                         )
mDEPLOhy.hd   = Set(initialize=mDEPLOhy.gd           , ordered=False, doc='retirement hs    units', filter=lambda mDEPLOhy,gd  : gd     in mDEPLOhy.gd            and  pH2RatedMaxStorage[gd] != 0.0                                         )
mDEPLOhy.rd   = Set(initialize=mDEPLOhy.gd           , ordered=False, doc='retirem   not st units', filter=lambda mDEPLOhy,gd  : gd     in mDEPLOhy.gd            and  pRatedMaxCharge   [gd] == 0.0 and pH2RatedMaxStorage[gd] == 0.0       )
mDEPLOhy.br   = Set(initialize=sBrList               , ordered=False, doc='all input     branches'                                                                                                                                           )
mDEPLOhy.la   = Set(initialize=dfNetwork.index       , ordered=False, doc='all input        lines'                                                                                                                                           )
mDEPLOhy.ndb  = Set(initialize=mDEPLOhy.nd           , ordered=True , doc='connection buy    node', filter=lambda mDEPLOhy,nd  : nd     in mDEPLOhy.nd            and  pECI         [nd].sum() > 0.0                                         )
mDEPLOhy.nds  = Set(initialize=mDEPLOhy.nd           , ordered=True , doc='connection sell   node', filter=lambda mDEPLOhy,nd  : nd     in mDEPLOhy.nd            and  pSpotPrice   [nd].sum() > 0.0                                         )
mDEPLOhy.ndp  = Set(initialize=mDEPLOhy.nd           , ordered=True , doc='PPA        buy    node', filter=lambda mDEPLOhy,nd  : nd     in mDEPLOhy.nd            and  any((nd,g) in mDEPLOhy.n2g and g in mDEPLOhy.ppa for g in mDEPLOhy.gg))

mDEPLOhy.co   = mDEPLOhy.el | mDEPLOhy.bp   # consumption-only units
mDEPLOhy.su   = mDEPLOhy.es - mDEPLOhy.co   # ESS storage, not consumption-only units (BESS)
mDEPLOhy.nr   = mDEPLOhy.el | mDEPLOhy.su   # units that can contribute to the operating reserves (BESS + electrolyzer)

# Instrumental sets
mDEPLOhy.sn     = [(  sc,n         ) for   sc,n          in mDEPLOhy.sc    * mDEPLOhy.n  ]
mDEPLOhy.sm     = [(  sc,m         ) for   sc,m          in mDEPLOhy.sc    * mDEPLOhy.m  ]
mDEPLOhy.sng    = [(  sc,n,g       ) for   sc,n,g        in mDEPLOhy.sn    * mDEPLOhy.g  ]
mDEPLOhy.snel   = [(  sc,n,el      ) for   sc,n,el       in mDEPLOhy.sn    * mDEPLOhy.el ]
mDEPLOhy.snhs   = [(  sc,n,hs      ) for   sc,n,hs       in mDEPLOhy.sn    * mDEPLOhy.hs ]
mDEPLOhy.snp    = [(  sc,n,ppa     ) for   sc,n,ppa      in mDEPLOhy.sn    * mDEPLOhy.ppa]
mDEPLOhy.snnr   = [(  sc,n,nr      ) for   sc,n,nr       in mDEPLOhy.sn    * mDEPLOhy.nr ]
mDEPLOhy.snd    = [(  sc,n,nd      ) for   sc,n,nd       in mDEPLOhy.sn    * mDEPLOhy.nd ]
mDEPLOhy.sndb   = [(  sc,n,ndb     ) for   sc,n,ndb      in mDEPLOhy.sn    * mDEPLOhy.ndb]
mDEPLOhy.snds   = [(  sc,n,nds     ) for   sc,n,nds      in mDEPLOhy.sn    * mDEPLOhy.nds]
mDEPLOhy.psnr   = [(p,sc,nr        ) for p,sc,nr         in mDEPLOhy.ps    * mDEPLOhy.nr ]
mDEPLOhy.pses   = [(p,sc,es        ) for p,sc,es         in mDEPLOhy.ps    * mDEPLOhy.es ]
mDEPLOhy.pssu   = [(p,sc,su        ) for p,sc,su         in mDEPLOhy.ps    * mDEPLOhy.su ]
mDEPLOhy.pshs   = [(p,sc,hs        ) for p,sc,hs         in mDEPLOhy.ps    * mDEPLOhy.hs ]
mDEPLOhy.psm    = [(p,sc,m         ) for p,sc,m          in mDEPLOhy.ps    * mDEPLOhy.m  ]
mDEPLOhy.psn    = [(p,sc,n         ) for p,sc,n          in mDEPLOhy.ps    * mDEPLOhy.n  ]
mDEPLOhy.psel   = [(p,sc,el        ) for p,sc,el         in mDEPLOhy.ps    * mDEPLOhy.el ]
mDEPLOhy.psppa  = [(p,sc,ppa       ) for p,sc,ppa        in mDEPLOhy.ps    * mDEPLOhy.ppa]
mDEPLOhy.psnm   = [(p,sc,n,m       ) for p,sc,n,m        in mDEPLOhy.psn   * mDEPLOhy.m  ]
mDEPLOhy.psng   = [(p,sc,n,g       ) for p,sc,n,g        in mDEPLOhy.psn   * mDEPLOhy.g  ]
mDEPLOhy.psngc  = [(p,sc,n,gc      ) for p,sc,n,gc       in mDEPLOhy.psn   * mDEPLOhy.gc ]
mDEPLOhy.psnppa = [(p,sc,n,ppa     ) for p,sc,n,ppa      in mDEPLOhy.psn   * mDEPLOhy.ppa]
mDEPLOhy.psnre  = [(p,sc,n,re      ) for p,sc,n,re       in mDEPLOhy.psn   * mDEPLOhy.re ]
mDEPLOhy.psnnr  = [(p,sc,n,nr      ) for p,sc,n,nr       in mDEPLOhy.psn   * mDEPLOhy.nr ]
mDEPLOhy.psnsu  = [(p,sc,n,su      ) for p,sc,n,su       in mDEPLOhy.psn   * mDEPLOhy.su ]
mDEPLOhy.psngp  = [(p,sc,n,gp      ) for p,sc,n,gp       in mDEPLOhy.psn   * mDEPLOhy.gp ]
mDEPLOhy.psnbp  = [(p,sc,n,bp      ) for p,sc,n,bp       in mDEPLOhy.psn   * mDEPLOhy.bp ]
mDEPLOhy.psnes  = [(p,sc,n,es      ) for p,sc,n,es       in mDEPLOhy.psn   * mDEPLOhy.es ]
mDEPLOhy.psnhs  = [(p,sc,n,hs      ) for p,sc,n,hs       in mDEPLOhy.psn   * mDEPLOhy.hs ]
mDEPLOhy.psnec  = [(p,sc,n,ec      ) for p,sc,n,ec       in mDEPLOhy.psn   * mDEPLOhy.ec ]
mDEPLOhy.psnhc  = [(p,sc,n,hc      ) for p,sc,n,hc       in mDEPLOhy.psn   * mDEPLOhy.hc ]
mDEPLOhy.psnnd  = [(p,sc,n,nd      ) for p,sc,n,nd       in mDEPLOhy.psn   * mDEPLOhy.nd ]
mDEPLOhy.psnnb  = [(p,sc,n,ndb     ) for p,sc,n,ndb      in mDEPLOhy.psn   * mDEPLOhy.ndb]
mDEPLOhy.psnns  = [(p,sc,n,nds     ) for p,sc,n,nds      in mDEPLOhy.psn   * mDEPLOhy.nds]
mDEPLOhy.psnel  = [(p,sc,n,el      ) for p,sc,n,el       in mDEPLOhy.psn   * mDEPLOhy.el ]
mDEPLOhy.psngd  = [(p,sc,n,gd      ) for p,sc,n,gd       in mDEPLOhy.psn   * mDEPLOhy.gd ]
mDEPLOhy.psned  = [(p,sc,n,ed      ) for p,sc,n,ed       in mDEPLOhy.psn   * mDEPLOhy.ed ]
mDEPLOhy.psnhd  = [(p,sc,n,hd      ) for p,sc,n,hd       in mDEPLOhy.psn   * mDEPLOhy.hd ]
mDEPLOhy.psnla  = [(p,sc,n,ni,nf,cc) for p,sc,n,ni,nf,cc in mDEPLOhy.psn   * mDEPLOhy.la ]
mDEPLOhy.psla   = [(p,sc,  ni,nf,cc) for p,sc,  ni,nf,cc in mDEPLOhy.ps    * mDEPLOhy.la ]
mDEPLOhy.pn     = [(p,n            ) for p,n             in mDEPLOhy.p     * mDEPLOhy.n  ]
mDEPLOhy.pnel   = [(p,   n,el      ) for p,   n,el       in mDEPLOhy.pn    * mDEPLOhy.el ]
mDEPLOhy.pg     = [(p,g            ) for p,g             in mDEPLOhy.p     * mDEPLOhy.g  ]
mDEPLOhy.pgc    = [(p,gc           ) for p,gc            in mDEPLOhy.p     * mDEPLOhy.gc ]
mDEPLOhy.pppa   = [(p,ppa          ) for p,ppa           in mDEPLOhy.p     * mDEPLOhy.ppa]
mDEPLOhy.pnr    = [(p,nr           ) for p,nr            in mDEPLOhy.p     * mDEPLOhy.nr ]
mDEPLOhy.pnm    = [(p,n,m          ) for p,n,m           in mDEPLOhy.pn    * mDEPLOhy.m  ]
mDEPLOhy.pel    = [(p,el           ) for p,el            in mDEPLOhy.p     * mDEPLOhy.el ]
mDEPLOhy.pgd    = [(p,gd           ) for p,gd            in mDEPLOhy.p     * mDEPLOhy.gd ]
mDEPLOhy.pla    = [(p,     ni,nf,cc) for p,     ni,nf,cc in mDEPLOhy.p     * mDEPLOhy.la ]
mDEPLOhy.pec    = [(p,ec           ) for p,ec            in mDEPLOhy.p     * mDEPLOhy.ec ]
mDEPLOhy.phc    = [(p,hc           ) for p,hc            in mDEPLOhy.p     * mDEPLOhy.hc ]
mDEPLOhy.phd    = [(p,hd           ) for p,hd            in mDEPLOhy.p     * mDEPLOhy.hd ]
mDEPLOhy.pbp    = [(p,bp           ) for p,bp            in mDEPLOhy.p     * mDEPLOhy.bp ]
mDEPLOhy.pm     = [(p,m            ) for p,m             in mDEPLOhy.p     * mDEPLOhy.m  ]
mDEPLOhy.lel    = [( lr,   el      ) for  lr,   el       in mDEPLOhy.lr    * mDEPLOhy.el ]
mDEPLOhy.lse    = [(    ls,el      ) for     ls,el       in mDEPLOhy.ls    * mDEPLOhy.el ]
mDEPLOhy.lle    = [( lr,ls,el      ) for  lr,ls,el       in mDEPLOhy.lr    * mDEPLOhy.lse]
mDEPLOhy.pcl    = [(p,        cl   ) for p,        cl    in mDEPLOhy.p     * mDEPLOhy.cl ]

# replacing string values by numerical values
idxDict        = dict()
idxDict[0    ] = 0
idxDict[0.0  ] = 0
idxDict['No' ] = 0
idxDict['NO' ] = 0
idxDict['no' ] = 0
idxDict['N'  ] = 0
idxDict['n'  ] = 0
idxDict['Yes'] = 1
idxDict['YES'] = 1
idxDict['yes'] = 1
idxDict['Y'  ] = 1
idxDict['y'  ] = 1

pIndBinUnitInvest  = pIndBinUnitInvest.map(idxDict)
pIndBinUnitRetire  = pIndBinUnitRetire.map(idxDict)

# Getting the current year
pCurrentYear = datetime.date.today().year

# Discount factor
if pAnnualDiscRate == 0.0:
    pDiscountFactor = pd.Series(data=[                        pPeriodWeight[p]                                                                                          for p  in mDEPLOhy.p ], index=mDEPLOhy.p)
else:
    pDiscountFactor = pd.Series(data=[((1.0+pAnnualDiscRate)**pPeriodWeight[p]-1.0) / (pAnnualDiscRate*(1.0+pAnnualDiscRate)**(pPeriodWeight[p]-1+p-pEconomicBaseYear)) for p  in mDEPLOhy.p ], index=mDEPLOhy.p)

# Adiabatic compression [MWh/kgH2 or GWh/tonH2]
pCompressionFactor = pd.Series(data=[((pHeatCapacity[bp] * (pInletT[bp] + 273) / (pIsEfficiency[bp] * pMechEfficiency[bp])) * (((pOutletP[bp] / pInletP[bp]) ** ((pGamma[bp] - 1) / pGamma[bp])) - 1) * (0.000001 / 3600)) for bp in mDEPLOhy.bp], index=mDEPLOhy.bp)

# Minimum and maximum variable power, charge, and storage capacity
pMaxPower     = pVarMaxPower     * pRatedMaxPower
pMinCharge    =                    pRatedMinCharge
pMaxCharge    = pVarMaxCharge    * pRatedMaxCharge
pMinOutflows  = pVarMinOutflows  * pH2RatedMinOutflow
pMaxOutflows  = pVarMaxOutflows  * pH2RatedMaxOutflow
pMinStorage   = pVarMinStorage   * pRatedMinStorage
pMaxStorage   = pVarMaxStorage   * pRatedMaxStorage
pH2MinStorage = pVarMinH2Storage * pH2RatedMinStorage
pH2MaxStorage = pVarMaxH2Storage * pH2RatedMaxStorage

pMaxPower     = pMaxPower.replace    (0.0, pRatedMaxPower    )
pMaxCharge    = pMaxCharge.replace   (0.0, pRatedMaxCharge   )
pMinOutflows  = pMinOutflows.replace (0.0, pH2RatedMinOutflow)
pMaxOutflows  = pMaxOutflows.replace (0.0, pH2RatedMaxOutflow)
pMinStorage   = pMinStorage.replace  (0.0, pRatedMinStorage  )
pMaxStorage   = pMaxStorage.replace  (0.0, pRatedMaxStorage  )
pH2MinStorage = pH2MinStorage.replace(0.0, pH2RatedMinStorage)
pH2MaxStorage = pH2MaxStorage.replace(0.0, pH2RatedMaxStorage)
pEfficiency   = pEfficiency.replace  (0.0, 1.0               )

pMaxPower     = pMaxPower.where    (pMaxPower     > 0.0, other=0.0)
pMinCharge    = pMinCharge.where   (pMinCharge    > 0.0, other=0.0)
pMaxCharge    = pMaxCharge.where   (pMaxCharge    > 0.0, other=0.0)
pMinOutflows  = pMinOutflows.where (pMinOutflows  > 0.0, other=0.0)
pMaxOutflows  = pMaxOutflows.where (pMaxOutflows  > 0.0, other=0.0)
pMinStorage   = pMinStorage.where  (pMinStorage   > 0.0, other=0.0)
pMaxStorage   = pMaxStorage.where  (pMaxStorage   > 0.0, other=0.0)
pH2MinStorage = pH2MinStorage.where(pH2MinStorage > 0.0, other=0.0)
pH2MaxStorage = pH2MaxStorage.where(pH2MaxStorage > 0.0, other=0.0)

# Initial inventory must be between minimum and maximum
pInitialInventory   = pInitialInventory.where  (pInitialInventory   > pRatedMinStorage  , pRatedMinStorage  )
pInitialInventory   = pInitialInventory.where  (pInitialInventory   < pRatedMaxStorage  , pRatedMaxStorage  )
pH2InitialInventory = pH2InitialInventory.where(pH2InitialInventory > pH2RatedMinStorage, pH2RatedMinStorage)
pH2InitialInventory = pH2InitialInventory.where(pH2InitialInventory < pH2RatedMaxStorage, pH2RatedMaxStorage)

# Drop levels with duration 0
pDuration            = pDuration.loc           [mDEPLOhy.n  ]
pSpotPrice           = pSpotPrice.loc          [mDEPLOhy.psn]
pResUpPrice          = pResUpPrice             [mDEPLOhy.psn]
pResDwPrice          = pResDwPrice             [mDEPLOhy.psn]
pEnergyUpPrice       = pEnergyUpPrice          [mDEPLOhy.psn]
pEnergyDwPrice       = pEnergyDwPrice          [mDEPLOhy.psn]
pActivationUp        = pActivationUp           [mDEPLOhy.psn]
pActivationDw        = pActivationDw           [mDEPLOhy.psn]
pMaxRatioDwUp        = pMaxRatioDwUp           [mDEPLOhy.psn]
pECI                 = pECI.loc                [mDEPLOhy.psn]
pMaxPower            = pMaxPower.loc           [mDEPLOhy.psn]
pMaxCharge           = pMaxCharge.loc          [mDEPLOhy.psn]
pMinOutflows         = pMinOutflows.loc        [mDEPLOhy.psn]
pMaxOutflows         = pMaxOutflows.loc        [mDEPLOhy.psn]
pMinStorage          = pMinStorage.loc         [mDEPLOhy.psn]
pMaxStorage          = pMaxStorage.loc         [mDEPLOhy.psn]
pH2MinStorage        = pH2MinStorage.loc       [mDEPLOhy.psn]
pH2MaxStorage        = pH2MaxStorage.loc       [mDEPLOhy.psn]
pMinCharge           = pd.DataFrame           ([pMinCharge  ] * len(mDEPLOhy.psn), index=pd.MultiIndex.from_tuples(mDEPLOhy.psn))
pPPAPrice            = pd.DataFrame           ([pPPAPrice   ] * len(mDEPLOhy.psn), index=pd.MultiIndex.from_tuples(mDEPLOhy.psn))

# Values < 1e-4 are converted to 0
pEpsilon   = 1e-4

# These parameters are in GW
pMaxPower           [pMaxPower            < pEpsilon] = 0.0
pMinCharge          [pMinCharge           < pEpsilon] = 0.0
pMaxCharge          [pMaxCharge           < pEpsilon] = 0.0

# These parameters are in GWh
pMinStorage         [pMinStorage          < pEpsilon] = 0.0
pMaxStorage         [pMaxStorage          < pEpsilon] = 0.0

# These parameters are in tonH2
pMinOutflows        [pMinOutflows         < pEpsilon] = 0.0
pMaxOutflows        [pMaxOutflows         < pEpsilon] = 0.0
pH2MinStorage       [pH2MinStorage        < pEpsilon] = 0.0
pH2MaxStorage       [pH2MaxStorage        < pEpsilon] = 0.0

# These parameters are in MEUR/MW
pInvestCost         [pInvestCost          < pEpsilon] = 0.0
pRetireCost         [pRetireCost          < pEpsilon] = 0.0

# These parameters are in MEUR/GWh
pSpotPrice          [pSpotPrice           < pEpsilon] = 0.0
pResUpPrice         [pResUpPrice          < pEpsilon] = 0.0
pResDwPrice         [pResDwPrice          < pEpsilon] = 0.0
pEnergyUpPrice      [pEnergyUpPrice       < pEpsilon] = 0.0
pEnergyDwPrice      [pEnergyDwPrice       < pEpsilon] = 0.0
pPPAPrice           [pPPAPrice            < pEpsilon] = 0.0


# RES and BESS degradation
if pIndDegradation == 1:
    pPeriodDegRate = pd.DataFrame([pPeriodDegRate] * len(mDEPLOhy.psn),      index=pd.MultiIndex.from_tuples(mDEPLOhy.psn))
    pDegIndex      = pd.Series([p - mDEPLOhy.p.first() for p in mDEPLOhy.p], index=list(mDEPLOhy.p))
    pDegFactor     = (1 - pPeriodDegRate)
    p_levels       = pDegFactor.index.get_level_values(0)
    aligned_powers = pDegIndex.reindex(p_levels).values
    pDegFactor     = pDegFactor.pow(aligned_powers, axis=0)

    for re in mDEPLOhy.re:
        pMaxPower[re]   = (pMaxPower[re]   * pDegFactor[re])
    for su in mDEPLOhy.su:
        pMaxStorage[su] = (pMaxStorage[su] * pDegFactor[su])

pMaxCharge2ndBlock   = pMaxCharge      - pMinCharge
pRatedMaxC2ndBlock   = pRatedMaxCharge - pRatedMinCharge
pMaxCharge2ndBlock    [pMaxCharge2ndBlock < pEpsilon] = 0.0


# =========================
# EZ curve parameters
# =========================

# Partial load curve [GW], partial production curve [tonH2/h], and ratio between minimum and maximum rated EZ charge [p.u.]
pPartialLoad     = pd.Series(data=[(lr * pRatedMaxCharge[el]               ) for lr,ls,el in mDEPLOhy.lle], index=mDEPLOhy.lle)
pPartialProd     = pd.Series(data=[(pPartialLoad[lr,ls,el] / pSE[lr,ls,el] ) for lr,ls,el in mDEPLOhy.lle], index=mDEPLOhy.lle)
pMinRatio        = pd.Series(data=[(pRatedMinCharge[el]/pRatedMaxCharge[el]) for       el in mDEPLOhy.el ], index=mDEPLOhy.el ).round(2)

# Finding the lr value closest to pMinRatio[el] to extract minimum rated H2 production [tH2/h]
pMinH2Prod            = pd.Series(index=mDEPLOhy.lse, dtype=float)
for ls,el in mDEPLOhy.lse:
    lr_values         = [lr for lr, el2 in mDEPLOhy.lel if el2 == el]
    lr_selected       = min(lr_values, key=lambda lr: abs(lr - pMinRatio[el]))
    pMinH2Prod[ls,el] = pPartialProd[(lr_selected,ls,el)].round(6)

# Filter only elements of lr ≥ pMinRatio
lr_valid = [lr for lr in mDEPLOhy.lr if lr >= pMinRatio[el]]

# Partial charge 2nd Block [GW] and partial production 2nd Block [tH2/h]
pPCharge2ndBlock   = pd.Series(data=[pPartialLoad[(lr,ls,el)] - pRatedMinCharge[el]   for lr,ls,el in mDEPLOhy.lle if lr in lr_valid], index=pd.MultiIndex.from_tuples([(lr,ls,el) for lr,ls,el in mDEPLOhy.lle if lr in lr_valid], names=['lr','ls','el'])).round(6)
pPProd2ndBlock     = pd.Series(data=[pPartialProd[(lr,ls,el)] - pMinH2Prod[ls,el]     for lr,ls,el in mDEPLOhy.lle if lr in lr_valid], index=pd.MultiIndex.from_tuples([(lr,ls,el) for lr,ls,el in mDEPLOhy.lle if lr in lr_valid], names=['lr','ls','el'])).round(6)

# Maximum rated H2 production [tonH2/h], and maximum H2 production 2nd Block [tonH2/h]
pMaxH2Prod         = pd.Series(data=[ pMinH2Prod[ls,el] + pPProd2ndBlock[(1.0,ls,el)] for ls,el in mDEPLOhy.lse], index=mDEPLOhy.lse)
pMaxH2Prod2ndBlock = pd.Series(data=[(pMaxH2Prod[ls,el] - pMinH2Prod[ls,el])          for ls,el in mDEPLOhy.lse], index=mDEPLOhy.lse)

# Linear approximation
if pIndPiecewise == 0:

    # Linear approximation (pH2ProdRate in [tonH2/GWh])
    lr_sel_map  = {el: min([lr for lr in mDEPLOhy.lr if lr >= pMinRatio[el]], key=lambda lr: abs(lr - pMinRatio[el])) for el in mDEPLOhy.el}
    X_first     = pd.Series(data=[pPCharge2ndBlock[(lr_sel_map[el],ls,el)]                              for    ls,el in mDEPLOhy.lse], index=pd.Index(mDEPLOhy.lse))
    X_last      = pd.Series(data=[pPCharge2ndBlock[(1.0           ,ls,el)]                              for    ls,el in mDEPLOhy.lse], index=pd.Index(mDEPLOhy.lse))
    Y_first     = pd.Series(data=[pPProd2ndBlock  [(lr_sel_map[el],ls,el)]                              for    ls,el in mDEPLOhy.lse], index=pd.Index(mDEPLOhy.lse))
    Y_last      = pd.Series(data=[pPProd2ndBlock  [(1.0           ,ls,el)]                              for    ls,el in mDEPLOhy.lse], index=pd.Index(mDEPLOhy.lse))
    pH2ProdRate = pd.Series(data=[((Y_last[ls,el] - Y_first[ls,el]) / (X_last[ls,el] - X_first[ls,el])) for    ls,el in mDEPLOhy.lse], index=pd.Index(mDEPLOhy.lse))
    Y_calc      = pd.Series(data=[pPCharge2ndBlock[lr,ls,el] * pH2ProdRate[ls,el]                       for lr,ls,el in mDEPLOhy.lle if lr in lr_valid], index=[(lr,ls,el) for lr,ls,el in mDEPLOhy.lle if lr in lr_valid])

    for ls,el in mDEPLOhy.lse:
        # Extract data vectors (a in [kgH2/kWh])
        X     = np.array([pPCharge2ndBlock[(lr,ls,el)] * 1e3 for lr in lr_valid], dtype=float)
        Y     = np.array([pPProd2ndBlock  [(lr,ls,el)] * 1e3 for lr in lr_valid], dtype=float)
        Y_fit = np.array([Y_calc          [(lr,ls,el)] * 1e3 for lr in lr_valid], dtype=float)
        a     = float   ( pH2ProdRate     [(   ls,el)]                         )

        # Plot manufacturer curve vs linear approximation
        if pIndFigures == 1:
            plt.figure()
            plt.plot(X,Y,     marker='o',     linestyle= '-', linewidth=1.0, label='Manufacturer curve')
            plt.plot(X,Y_fit, color ='black', linestyle='--', linewidth=0.5, label=f'Linear approx: Y = {a:.3f}[kgH2/MWh]·X')
            plt.xlabel('Charge2ndBlock [MW]')
            plt.ylabel('H2Prod2ndBlock [kgH2/h]')
            plt.title(f'{el}: {ls} curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(_path + f'/DEPLOhy_Plot_LinearFit_{el}_{ls}_' + CaseName + '.png', dpi=300)
            plt.close()

# Piecewise approximation
if pIndPiecewise == 1:

    # Number of points per segment boundary
    n_L = 6  # first 6 points in low  range
    n_H = 6  # last  6 points in high range

    # Containers for segmented data
    X_seg          = {}  # X_seg[(ls,el,cl)] = list of X values for segment cl
    Y_seg          = {}  # Y_seg[(ls,el,cl)] = list of Y values for segment cl
    pSegH2ProdRate = {}  # pSegH2ProdRate[(ls,el,cl)] = slope of linear fit
    pSegIntercept  = {}  # intercept for each segment
    Y_fit_seg      = {}  # linear approximation values aligned with X_seg

    for ls,el in mDEPLOhy.lse:
        # Extract full manufacturer curve for this (ls,el)
        X     = np.array([pPCharge2ndBlock[(lr,ls,el)] * 1e3 for lr in lr_valid], dtype=float)
        Y     = np.array([pPProd2ndBlock  [(lr,ls,el)] * 1e3 for lr in lr_valid], dtype=float)
        n     = len(X)

        # range segments: first 6 points for LOW, last 6 points for HIGH, middle (overlapping with last point of L and first point of H) for MIDDLE
        idx_L = np.arange(0      , n_L          )
        idx_H = np.arange(n - n_H, n            )
        idx_M = np.arange(n_L - 1, n - (n_H - 1))

        # Assign segments
        X_seg[(ls,el,mDEPLOhy.cl.first() )] = X[idx_L]
        Y_seg[(ls,el,mDEPLOhy.cl.first() )] = Y[idx_L]

        X_seg[(ls,el,list(mDEPLOhy.cl)[1])] = X[idx_M]
        Y_seg[(ls,el,list(mDEPLOhy.cl)[1])] = Y[idx_M]

        X_seg[(ls,el,mDEPLOhy.cl.last()  )] = X[idx_H]
        Y_seg[(ls,el,mDEPLOhy.cl.last()  )] = Y[idx_H]

        # Linear approximations for each segment
        for cl in mDEPLOhy.cl:
            Xs = X_seg[(ls,el,cl)]
            Ys = Y_seg[(ls,el,cl)]

            # Slope defined between first and last point [MWh/kgH2] = [GWh/tonH2]
            slope = (Ys[-1] - Ys[0]) / (Xs[-1] - Xs[0])
            pSegH2ProdRate[(ls,el,cl)] = slope.round(4)

            # Intercept [tonH2/h]
            intercept = Ys[0] - (slope * Xs[0])
            pSegIntercept[(ls,el,cl)] = (intercept * 1e-3).round(7)

            # Linear fit over segment
            Y_fit = intercept + (slope * Xs)
            Y_fit_seg[(ls,el,cl)] = Y_fit

    # Plot manufacturer curve vs piecewise linear fits
    if pIndFigures == 1:
        for ls,el in mDEPLOhy.lse:
            plt.figure()

            # Plot full manufacturer curve (all points)
            X_full = np.array([pPCharge2ndBlock[(lr,ls,el)] * 1e3 for lr in lr_valid], dtype=float)
            Y_full = np.array([pPProd2ndBlock  [(lr,ls,el)] * 1e3 for lr in lr_valid], dtype=float)
            plt.plot(X_full, Y_full, marker='o', linestyle= '-', linewidth=1.0, label='Manufacturer curve')

            # Plot linear fits for each segment (L,M,H)
            for cl in mDEPLOhy.cl:
                Xs   = X_seg               [(ls,el,cl)]
                Yfit = Y_fit_seg           [(ls,el,cl)]
                a    = float(pSegH2ProdRate[(ls,el,cl)]      )
                b    = float(pSegIntercept [(ls,el,cl)] * 1e3)
                plt.plot(Xs, Yfit,               linestyle='--', linewidth=0.5, label=f'Piecewise fit ({cl}): Y = {a:.3f}[kgH2/MWh]·X + {b:.3f}[kgH2]')

            # Plot styling
            plt.xlabel('Charge2ndBlock [MW]')
            plt.ylabel('H2Prod2ndBlock [kgH2/h]')
            plt.title(f'{el}: {ls} curve')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(_path + f'/DEPLOhy_Plot_PiecewiseFit_{el}_{ls}_' + CaseName + '.png', dpi=300)
            plt.close()

# MaxProduction among all periods [tonH2/h]
pMaxH2Inflow = pMaxH2Prod.max()

# pRatedMaxCharge among all electrolyzers [GW]
pEZRatedMaxCharge = pRatedMaxCharge.loc[list(mDEPLOhy.el)].max()


# Re-indexing electrolyzer parameters: p if linear, and (p,cl) if piecewise
def map_series_lse_to_p(series_lse,p2l):
    """
    Maps a pd.Series indexed by (ls,el) → p, keeping only entries defined in p2l.
    """
    data = {p: series_lse[(ls,el)] for (p,el,ls) in p2l if (ls,el) in series_lse.index}
    return pd.Series(data)

pMinH2Prod_p         = map_series_lse_to_p(pMinH2Prod        ,mDEPLOhy.p2l)
pMaxH2Prod_p         = map_series_lse_to_p(pMaxH2Prod        ,mDEPLOhy.p2l)
pMaxH2Prod2ndBlock_p = map_series_lse_to_p(pMaxH2Prod2ndBlock,mDEPLOhy.p2l)

pMinH2Prod_p         = pMinH2Prod_p        .reindex(mDEPLOhy.p)
pMaxH2Prod_p         = pMaxH2Prod_p        .reindex(mDEPLOhy.p)
pMaxH2Prod2ndBlock_p = pMaxH2Prod2ndBlock_p.reindex(mDEPLOhy.p)

if pIndPiecewise == 1:
    def map_series_lle_to_pcl(series_lle,p2l):
        """
        Maps a pd.Series indexed by (ls,el,cl) → (p,cl), using p2l = (p,el,ls). Keeps only valid mappings defined in p2l.
        """
        data = {(p,cl): series_lle[(ls,el,cl)] for (p,el,ls) in p2l for (ls2,el2,cl) in series_lle.keys() if ls2 == ls and el2 == el}
        return pd.Series(data)

    pSegH2ProdRate_pcl = map_series_lle_to_pcl(pSegH2ProdRate,mDEPLOhy.p2l)
    pSegIntercept_pcl  = map_series_lle_to_pcl(pSegIntercept, mDEPLOhy.p2l)

    pSegH2ProdRate_pcl = pSegH2ProdRate_pcl.reindex(mDEPLOhy.pcl)
    pSegIntercept_pcl  = pSegIntercept_pcl .reindex(mDEPLOhy.pcl)

else:
    pH2ProdRate_p      = map_series_lse_to_p(pH2ProdRate, mDEPLOhy.p2l)
    pH2ProdRate_p      = pH2ProdRate_p.reindex(mDEPLOhy.p)

# Replace < 0.0 by 0.0
pMaxCharge2ndBlock = pMaxCharge2ndBlock.where(pMaxCharge2ndBlock > 0.0, 0.0)

# Grid electricity cost [MEUR/GWh] and PPA cost [MEUR/GWh]
pEnergyTerm   = pEnergyTerm.to_dict()
pMarketPeriod = pMarketPeriod.stack().to_dict()
pSpotPrice    = pSpotPrice.loc[:,mDEPLOhy.nds]
pSpotPrice    = pSpotPrice.stack().to_dict()
pPPAPrice     = pPPAPrice.stack().to_dict()
pEnergyTariff = pd.Series(data=[sum(pEnergyTerm[p,m] * pMarketPeriod[p,n,m] for m in mDEPLOhy.m) for p,n        in mDEPLOhy.pn    ], index=[(p,n)        for p,n        in mDEPLOhy.pn    ])
pEnergyCost   = pd.Series(data=[pEnergyTariff  [p,n] + pSpotPrice   [p,sc,n,nds]                 for p,sc,n,nds in mDEPLOhy.psnns ], index=[(p,sc,n,ndb) for p,sc,n,ndb in mDEPLOhy.psnnb ])
pPPACost      = pd.Series(data=[pEnergyTariff  [p,n] + pPPAPrice    [p,sc,n,ppa]                 for p,sc,n,ppa in mDEPLOhy.psnppa], index=[(p,sc,n,ppa) for p,sc,n,ppa in mDEPLOhy.psnppa])
pMarketPeriod = pd.Series(data=[pMarketPeriod[p,n,m]                                             for p,n,m      in mDEPLOhy.pnm   ], index=[(p,n,m)      for p,n,m      in mDEPLOhy.pnm   ])

# Drop units not g
pInitialPeriod      = pInitialPeriod.loc     [   mDEPLOhy.g ]
pFinalPeriod        = pFinalPeriod.loc       [   mDEPLOhy.g ]
pRatedMaxPower      = pRatedMaxPower.loc     [   mDEPLOhy.g ]
pRatedMaxCharge     = pRatedMaxCharge.loc    [   mDEPLOhy.g ]
pOMCost             = pOMCost.loc            [   mDEPLOhy.g ]
pMaxPower           = pMaxPower.loc          [:, mDEPLOhy.g ]
pIndBinUnitInvest   = pIndBinUnitInvest.loc  [   mDEPLOhy.g ]
pLowerInvest        = pLowerInvest.loc       [   mDEPLOhy.g ]
pUpperInvest        = pUpperInvest.loc       [   mDEPLOhy.g ]

# Drop units not st
pEfficiency         = pEfficiency.loc        [   mDEPLOhy.st]

# Drop units not es
pMaxCharge          = pMaxCharge.loc         [:, mDEPLOhy.es]

# Drop units not su
pMinStorage         = pMinStorage.loc        [:, mDEPLOhy.su]
pMaxStorage         = pMaxStorage.loc        [:, mDEPLOhy.su]
pInitialInventory   = pInitialInventory.loc  [   mDEPLOhy.su]
pRatedMaxStorage    = pInitialInventory.loc  [   mDEPLOhy.su]

# Drop units not hs
pH2RatedMinOutflow  = pH2RatedMinOutflow.loc [   mDEPLOhy.hs]
pH2RatedMaxOutflow  = pH2RatedMaxOutflow.loc [   mDEPLOhy.hs]
pMinOutflows        = pMinOutflows.loc       [:, mDEPLOhy.hs]
pMaxOutflows        = pMaxOutflows.loc       [:, mDEPLOhy.hs]
pH2RatedMaxStorage  = pH2RatedMaxStorage.loc [   mDEPLOhy.hs]
pH2MinStorage       = pH2MinStorage.loc      [:, mDEPLOhy.hs]
pH2MaxStorage       = pH2MaxStorage.loc      [:, mDEPLOhy.hs]
pH2InitialInventory = pH2InitialInventory.loc[   mDEPLOhy.hs]
pOutflowsRampUp     = pOutflowsRampUp.loc    [   mDEPLOhy.hs]
pOutflowsRampDw     = pOutflowsRampDw.loc    [   mDEPLOhy.hs]

# Drop units not gc or gd
pInvestCost         = pInvestCost.loc        [   mDEPLOhy.gc]
pRetireCost         = pRetireCost.loc        [   mDEPLOhy.gd]
pIndBinUnitRetire   = pIndBinUnitRetire.loc  [   mDEPLOhy.gd]
pLowerRetire        = pLowerRetire.loc       [   mDEPLOhy.gd]
pUpperRetire        = pUpperRetire.loc       [   mDEPLOhy.gd]

# Drop units not el
pMinCharge          = pMinCharge.loc         [:, mDEPLOhy.el]
pStartUpCost        = pStartUpCost.loc       [   mDEPLOhy.el]
pShutDownCost       = pShutDownCost.loc      [   mDEPLOhy.el]

# Drop units not bp
pSBConsumption      = pSBConsumption.loc     [   mDEPLOhy.bp]
pBoPFactor          = pBoPFactor.loc         [   mDEPLOhy.bp]

# Drop units not nr
pMaxCharge2ndBlock  = pMaxCharge2ndBlock.loc [:, mDEPLOhy.nr]

# Drop nodes that are not explicitly grid-connected
pECI                = pECI.loc               [:,mDEPLOhy.ndb]
pECI                = pECI.stack().to_dict()


# Max grid connection capacity [GW] depending on hydrogen certification mechanism
# using 120 MJ/kgH2 as LHV
if pIndLCF   == 1:
    pMaxGCP_n   = pd.Series(data=[(1 - pSavingTarget) * pFossilComparator * pMinH2Prod_p[p] * 120                                    for p          in mDEPLOhy.p    ], index=mDEPLOhy.p    )
    pMaxGCP_d   = pd.Series(data=[pECI[p,sc,n,ndb] * 3600                                                                            for p,sc,n,ndb in mDEPLOhy.psnnb], index=mDEPLOhy.psnnb)
    pMaxGCP     = pd.Series(data=[(min((pMaxGCP_n[p] / pMaxGCP_d[p,sc,n,ndb]), pGCPCapacity) if pMaxGCP_d[p,sc,n,ndb] >  0 else 0.0) for p,sc,n,ndb in mDEPLOhy.psnnb], index=mDEPLOhy.psnnb).round(6)
    pMaxGCPzero = pd.Series(data=[                                            (pGCPCapacity  if pMaxGCP_d[p,sc,n,ndb] == 0 else 0.0) for p,sc,n,ndb in mDEPLOhy.psnnb], index=mDEPLOhy.psnnb)

# This option avoids a warning in the following assignments
pd.options.mode.chained_assignment = None


# =========================
# Parameters
# =========================

# General
mDEPLOhy.pPeriodWeight        = Param(mDEPLOhy.p     , initialize=pPeriodWeight.to_dict()               , within=NonNegativeIntegers, doc='Period weight'   , mutable=True)
mDEPLOhy.pMinAnnualOutflows   = Param(mDEPLOhy.p     , initialize=pMinAnnualOutflows.to_dict()          , within=NonNegativeIntegers, doc='Min yearly H2 production'      )
mDEPLOhy.pDiscountFactor      = Param(mDEPLOhy.p     , initialize=pDiscountFactor.to_dict()             , within=NonNegativeReals,    doc='Discount factor'               )
mDEPLOhy.pAPRE                = Param(mDEPLOhy.p     , initialize=pAPRE.to_dict()                       , within=NonNegativeReals,    doc='Avge proportion of ren electr.')
mDEPLOhy.pNRH                 = Param(mDEPLOhy.p     , initialize=pNRH.to_dict()                        , within=NonNegativeIntegers, doc='Nr of ren electricity hours'   )
mDEPLOhy.pScenProb            = Param(mDEPLOhy.ps    , initialize=pScenProb.to_dict()                   , within=UnitInterval,        doc='Probability'     , mutable=True)
mDEPLOhy.pDuration            = Param(mDEPLOhy.n     , initialize=pDuration.to_dict()                   , within=PositiveIntegers,    doc='Duration'        , mutable=True)
mDEPLOhy.pEnergyCost          = Param(mDEPLOhy.psnnb , initialize=pEnergyCost                           , within=           Reals,    doc='Power cost in buy  node'       )
mDEPLOhy.pSpotPrice           = Param(mDEPLOhy.psnns , initialize=pSpotPrice                            , within=           Reals,    doc='Spot price in sell node'       )
mDEPLOhy.pResUpPrice          = Param(mDEPLOhy.psn   , initialize=pResUpPrice.to_dict()                 , within=           Reals,    doc='Balancing capacity up price'   )
mDEPLOhy.pResDwPrice          = Param(mDEPLOhy.psn   , initialize=pResDwPrice.to_dict()                 , within=           Reals,    doc='Balancing capacity dw price'   )
mDEPLOhy.pEnergyUpPrice       = Param(mDEPLOhy.psn   , initialize=pEnergyUpPrice.to_dict()              , within=           Reals,    doc='Balancing energy   up price'   )
mDEPLOhy.pEnergyDwPrice       = Param(mDEPLOhy.psn   , initialize=pEnergyDwPrice.to_dict()              , within=           Reals,    doc='Balancing energy   dw price'   )
mDEPLOhy.pActivationUp        = Param(mDEPLOhy.psn   , initialize=pActivationUp.to_dict()               , within=UnitInterval,        doc='Activation requirement up'     )
mDEPLOhy.pActivationDw        = Param(mDEPLOhy.psn   , initialize=pActivationDw.to_dict()               , within=UnitInterval,        doc='Activation requirement dw'     )
mDEPLOhy.pMaxRatioDwUp        = Param(mDEPLOhy.psn   , initialize=pMaxRatioDwUp.to_dict()               , within=NonNegativeReals,    doc='Max ratio between up / dw cap' )
mDEPLOhy.pPPACost             = Param(mDEPLOhy.psnppa, initialize=pPPACost                              , within=NonNegativeReals,    doc='PPA cost including grid tariff')
mDEPLOhy.pMarketPeriod        = Param(mDEPLOhy.pnm   , initialize=pMarketPeriod                         , within=NonNegativeIntegers, doc='Hourly energy market period'   )
mDEPLOhy.pPowerTariff         = Param(mDEPLOhy.pm    , initialize=pPowerTariff.to_dict()                , within=NonNegativeReals,    doc='Power term grid tariff'        )
mDEPLOhy.pFixedPower          = Param(mDEPLOhy.pm    , initialize=pFixedPower.to_dict()                 , within=NonNegativeReals,    doc='Fixed contracted power'        )

# Parameters
mDEPLOhy.pTimeStep            = Param(                 initialize=pTimeStep                             , within=PositiveIntegers,    doc='Unitary time step'              )
mDEPLOhy.pAnnualDiscRate      = Param(                 initialize=pAnnualDiscRate                       , within=UnitInterval,        doc='Annual discount rate'           )
mDEPLOhy.pMinCapFactor        = Param(                 initialize=pMinCapFactor                         , within=UnitInterval,        doc='Annual EZ capacity factor'      )
mDEPLOhy.pGCPCapacity         = Param(                 initialize=pGCPCapacity                          , within=NonNegativeReals,    doc='Grid connection point capacity' )
mDEPLOhy.pVAT                 = Param(                 initialize=pVAT                                  , within=UnitInterval,        doc='Value added tax'                )
mDEPLOhy.pEnergyTax           = Param(                 initialize=pEnergyTax                            , within=UnitInterval,        doc='Electricity tax'                )
if pIndLCF == 1:
    mDEPLOhy.pMaxGCP          = Param(mDEPLOhy.psnnb , initialize=pMaxGCP.to_dict()                     , within=NonNegativeReals,    doc='Max CPC when ECI != 0'          )
    mDEPLOhy.pMaxGCPzero      = Param(mDEPLOhy.psnnb , initialize=pMaxGCPzero.to_dict()                 , within=NonNegativeReals,    doc='Max CPC when ECI == 0'          )

# Options
mDEPLOhy.pIndBinInvest        = Param(                 initialize=pIndBinInvest                         , within=NonNegativeIntegers, doc='Indicator of binary investment decisions'              , mutable=True)
mDEPLOhy.pIndBinRetire        = Param(                 initialize=pIndBinRetire                         , within=NonNegativeIntegers, doc='Indicator of binary retirement decisions'              , mutable=True)
mDEPLOhy.pIndBinOperation     = Param(                 initialize=pIndBinOperation                      , within=Binary,              doc='Indicator of binary operation  decisions'              , mutable=True)
mDEPLOhy.pIndReserves         = Param(                 initialize=pIndReserves                          , within=Binary,              doc='Indicator of balancing reserves provision'             , mutable=True)
mDEPLOhy.pIndPiecewise        = Param(                 initialize=pIndPiecewise                         , within=Binary,              doc='Indicator of electrolyzer curve piecewise linear fit'  , mutable=True)
mDEPLOhy.pIndTwoStates        = Param(                 initialize=pIndTwoStates                         , within=Binary,              doc='Indicator of two-state electrolyzer model (On-SB)'     , mutable=True)
mDEPLOhy.pIndLCF              = Param(                 initialize=pIndLCF                               , within=Binary,              doc='Indicator of compliance with low carbon fuel certific' , mutable=True)
mDEPLOhy.pIndRFNBO            = Param(                 initialize=pIndRFNBO                             , within=Binary,              doc='Indicator of compliance with RFNBO certification'      , mutable=True)
mDEPLOhy.pIndCapFactor        = Param(                 initialize=pIndCapFactor                         , within=Binary,              doc='Indicator of capacity factor constraint'               , mutable=True)

# Components
mDEPLOhy.pIndBinUnitInvest    = Param(mDEPLOhy.g     , initialize=pIndBinUnitInvest.to_dict()           , within=Binary,              doc='Binary investment decision'    )
mDEPLOhy.pIndBinUnitRetire    = Param(mDEPLOhy.gd    , initialize=pIndBinUnitRetire.to_dict()           , within=Binary,              doc='Binary retirement decision'    )
mDEPLOhy.pInitialPeriod       = Param(mDEPLOhy.g     , initialize=pInitialPeriod.to_dict()              , within=NonNegativeIntegers, doc='installation year'             )
mDEPLOhy.pFinalPeriod         = Param(mDEPLOhy.g     , initialize=pFinalPeriod.to_dict()                , within=NonNegativeIntegers, doc='retirement   year'             )
mDEPLOhy.pMaxPower            = Param(mDEPLOhy.psng  , initialize=pMaxPower.stack().to_dict()           , within=NonNegativeReals,    doc='Maximum power'                 )
mDEPLOhy.pMinCharge           = Param(mDEPLOhy.psnel , initialize=pMinCharge.stack().to_dict()          , within=NonNegativeReals,    doc='Minimum charge'                )
mDEPLOhy.pMaxCharge           = Param(mDEPLOhy.psnes , initialize=pMaxCharge.stack().to_dict()          , within=NonNegativeReals,    doc='Maximum charge'                )
mDEPLOhy.pMaxCharge2ndBlock   = Param(mDEPLOhy.psnnr , initialize=pMaxCharge2ndBlock.stack().to_dict()  , within=NonNegativeReals,    doc='Second block charge'           )
mDEPLOhy.pRatedMaxPower       = Param(mDEPLOhy.g     , initialize=pRatedMaxPower.to_dict()              , within=NonNegativeReals,    doc='Rated maximum power'           )
mDEPLOhy.pRatedMaxCharge      = Param(mDEPLOhy.g     , initialize=pRatedMaxCharge.to_dict()             , within=NonNegativeReals,    doc='Rated maximum charge'          )
mDEPLOhy.pRatedMaxStorage     = Param(mDEPLOhy.su    , initialize=pRatedMaxStorage.to_dict()            , within=NonNegativeReals,    doc='Rated BESS storage capacity'   )
mDEPLOhy.pInitialInventory    = Param(mDEPLOhy.su    , initialize=pInitialInventory.to_dict()           , within=NonNegativeReals,    doc='Initial BESS inventory'        )
mDEPLOhy.pMinStorage          = Param(mDEPLOhy.psnsu , initialize=pMinStorage.stack().to_dict()         , within=NonNegativeReals,    doc='Minimum BESS storage capacity' )
mDEPLOhy.pMaxStorage          = Param(mDEPLOhy.psnsu , initialize=pMaxStorage.stack().to_dict()         , within=NonNegativeReals,    doc='Maximum BESS storage capacity' )
mDEPLOhy.pH2RatedMaxStorage   = Param(mDEPLOhy.hs    , initialize=pH2RatedMaxStorage.to_dict()          , within=NonNegativeReals,    doc='Rated max h2 storage capacity' )
mDEPLOhy.pH2InitialInventory  = Param(mDEPLOhy.hs    , initialize=pH2InitialInventory.to_dict()         , within=NonNegativeReals,    doc='Initial   h2 storage inventory')
mDEPLOhy.pMinOutflows         = Param(mDEPLOhy.psnhs , initialize=pMinOutflows.stack().to_dict()        , within=NonNegativeReals,    doc='Minimum h2         outflows'   )
mDEPLOhy.pMaxOutflows         = Param(mDEPLOhy.psnhs , initialize=pMaxOutflows.stack().to_dict()        , within=NonNegativeReals,    doc='Maximum h2         outflows'   )
mDEPLOhy.pH2MinStorage        = Param(mDEPLOhy.psnhs , initialize=pH2MinStorage.stack().to_dict()       , within=NonNegativeReals,    doc='Minimum h2 storage capacity'   )
mDEPLOhy.pH2MaxStorage        = Param(mDEPLOhy.psnhs , initialize=pH2MaxStorage.stack().to_dict()       , within=NonNegativeReals,    doc='Maximum h2 storage capacity'   )
mDEPLOhy.pEfficiency          = Param(mDEPLOhy.st    , initialize=pEfficiency.to_dict()                 , within=UnitInterval,        doc='Round-trip efficiency'         )
mDEPLOhy.pMaxH2Prod           = Param(mDEPLOhy.p     , initialize=pMaxH2Prod_p.to_dict()                , within=NonNegativeReals,    doc='Max hydrogen production EZ'    )
mDEPLOhy.pMinH2Prod           = Param(mDEPLOhy.p     , initialize=pMinH2Prod_p.to_dict()                , within=NonNegativeReals,    doc='Min hydrogen production EZ'    )
mDEPLOhy.pMaxH2Prod2ndBlock   = Param(mDEPLOhy.p     , initialize=pMaxH2Prod2ndBlock_p.to_dict()        , within=NonNegativeReals,    doc='Max production above min value')
mDEPLOhy.pStartUpCost         = Param(mDEPLOhy.el    , initialize=pStartUpCost.to_dict()                , within=NonNegativeReals,    doc='EZ start-up cost'              )
mDEPLOhy.pShutDownCost        = Param(mDEPLOhy.el    , initialize=pShutDownCost.to_dict()               , within=NonNegativeReals,    doc='EZ shutdown cost'              )
mDEPLOhy.pOMCost              = Param(mDEPLOhy.g     , initialize=pOMCost.to_dict()                     , within=NonNegativeReals,    doc='O&M annual fixed  cost'        )
mDEPLOhy.pInvestCost          = Param(mDEPLOhy.gc    , initialize=pInvestCost.to_dict()                 , within=NonNegativeReals,    doc='Unit fixed investment cost'    )
mDEPLOhy.pRetireCost          = Param(mDEPLOhy.gd    , initialize=pRetireCost.to_dict()                 , within=NonNegativeReals,    doc='Unit fixed retirement cost'    )
mDEPLOhy.pLowerInvest         = Param(mDEPLOhy.g     , initialize=pLowerInvest.to_dict()                , within=NonNegativeReals,    doc='Lower investment bound',  mutable=True)
mDEPLOhy.pUpperInvest         = Param(mDEPLOhy.g     , initialize=pUpperInvest.to_dict()                , within=NonNegativeReals,    doc='Upper investment bound',  mutable=True)
mDEPLOhy.pLowerRetire         = Param(mDEPLOhy.gd    , initialize=pLowerRetire.to_dict()                , within=NonNegativeReals,    doc='Lower retirement bound',  mutable=True)
mDEPLOhy.pUpperRetire         = Param(mDEPLOhy.gd    , initialize=pUpperRetire.to_dict()                , within=NonNegativeReals,    doc='Upper retirement bound',  mutable=True)
mDEPLOhy.pOutflowsRampUp      = Param(mDEPLOhy.hs    , initialize=pOutflowsRampUp.to_dict()             , within=NonNegativeReals,    doc='H2 outflows ramp up rate'      )
mDEPLOhy.pOutflowsRampDw      = Param(mDEPLOhy.hs    , initialize=pOutflowsRampDw.to_dict()             , within=Reals,               doc='H2 outflows ramp down rate'    )
mDEPLOhy.pCompressionFactor   = Param(mDEPLOhy.bp    , initialize=pCompressionFactor.to_dict()          , within=NonNegativeReals,    doc='Compression factor'            )
mDEPLOhy.pBoPFactor           = Param(mDEPLOhy.bp    , initialize=pBoPFactor.to_dict()                  , within=UnitInterval,        doc='BoP factor'                    )
mDEPLOhy.pSBConsumption       = Param(mDEPLOhy.bp    , initialize=pSBConsumption.to_dict()              , within=NonNegativeReals,    doc='BoP consumption during standby')

if pIndPiecewise == 0:
    mDEPLOhy.pH2ProdRate      = Param(mDEPLOhy.p     , initialize=pH2ProdRate_p.to_dict()               , within=NonNegativeReals,    doc='EZ linear  production rate'    )
else:
    mDEPLOhy.pSegH2ProdRate   = Param(mDEPLOhy.pcl   , initialize=pSegH2ProdRate_pcl.to_dict()          , within=NonNegativeReals,    doc='EZ segment production rate'    )
    mDEPLOhy.pSegIntercept    = Param(mDEPLOhy.pcl   , initialize=pSegIntercept_pcl.to_dict()           , within=NonNegativeReals,    doc='EZ segment intercept'          )

# Operation
mDEPLOhy.pPeriodProb           = Param(mDEPLOhy.ps   , initialize=0.0                                   , within=NonNegativeReals,    doc='Period probability',      mutable=True)
mDEPLOhy.pMaxPeriodOutflows    = Param(mDEPLOhy.p    , initialize=0.0                                   , within=NonNegativeReals,    doc='Max     period outflows', mutable=True)
mDEPLOhy.pMinPeriodOutflows    = Param(mDEPLOhy.p    , initialize=0.0                                   , within=NonNegativeReals,    doc='Min     period outflows', mutable=True)
mDEPLOhy.pMaxTotalOutflows     = Param(                initialize=0.0                                   , within=NonNegativeReals,    doc='Max      total outflows', mutable=True)
mDEPLOhy.pMinTotalOutflows     = Param(                initialize=0.0                                   , within=NonNegativeReals,    doc='Min      total outflows', mutable=True)
mDEPLOhy.pMaxDiscTotalOutflows = Param(                initialize=0.0                                   , within=NonNegativeReals,    doc='Max disc total outflows', mutable=True)
mDEPLOhy.pMinDiscTotalOutflows = Param(                initialize=0.0                                   , within=NonNegativeReals,    doc='Min disc total outflows', mutable=True)

# Periods and scenarios are going to be solved together with their weight and probability
for p,sc in mDEPLOhy.ps:
    mDEPLOhy.pPeriodProb       [p,sc] = mDEPLOhy.pPeriodWeight[p] * mDEPLOhy.pScenProb[p,sc]

# Max and min H2 outflows (periodic, total)
PeriodScen = {p: [sc for pp,sc in mDEPLOhy.ps if pp == p] for p in mDEPLOhy.p}

for (p,sc) in mDEPLOhy.ps:
    mDEPLOhy.pMaxPeriodOutflows[p] = sum(mDEPLOhy.pMaxOutflows       [p,sc,n,hs]   * mDEPLOhy.pDuration[n]       * mDEPLOhy.pPeriodWeight[p]    for n in mDEPLOhy.n for hs in mDEPLOhy.hs)
    mDEPLOhy.pMinPeriodOutflows[p] = sum(mDEPLOhy.pMinOutflows       [p,sc,n,hs]   * mDEPLOhy.pDuration[n]       * mDEPLOhy.pPeriodWeight[p]    for n in mDEPLOhy.n for hs in mDEPLOhy.hs)

mDEPLOhy.pMaxTotalOutflows         = sum( mDEPLOhy.pMaxPeriodOutflows[p        ]()                                                              for p in mDEPLOhy.p)
mDEPLOhy.pMinTotalOutflows         = sum( mDEPLOhy.pMinPeriodOutflows[p        ]()                                                              for p in mDEPLOhy.p)
mDEPLOhy.pMaxDiscTotalOutflows     = sum((mDEPLOhy.pMaxPeriodOutflows[p        ]() * mDEPLOhy.pDiscountFactor[p] / mDEPLOhy.pPeriodWeight[p]()) for p in mDEPLOhy.p)
mDEPLOhy.pMinDiscTotalOutflows     = sum((mDEPLOhy.pMinPeriodOutflows[p        ]() * mDEPLOhy.pDiscountFactor[p] / mDEPLOhy.pPeriodWeight[p]()) for p in mDEPLOhy.p)


# =========================
# Variables
# =========================

mDEPLOhy.vUnitOpCost           = Var(mDEPLOhy.pg ,    within=NonNegativeReals,                                                                                                                doc='unit   fixed & variable     operation cost [MEUR]')
mDEPLOhy.vUnitICost            = Var(mDEPLOhy.pgc,    within=NonNegativeReals,                                                                                                                doc='unit   investment                     cost [MEUR]')
mDEPLOhy.vUnitRCost            = Var(mDEPLOhy.pgd,    within=NonNegativeReals,                                                                                                                doc='unit   retirement                     cost [MEUR]')
mDEPLOhy.vPeriodSCost          = Var(mDEPLOhy.p,      within=           Reals,                                                                                                                doc='period system                         cost [MEUR]')
mDEPLOhy.vPeriodFCost          = Var(mDEPLOhy.p,      within=NonNegativeReals,                                                                                                                doc='period system fixed capital           cost [MEUR]')
mDEPLOhy.vPeriodOpCost         = Var(mDEPLOhy.p,      within=NonNegativeReals,                                                                                                                doc='period system fixed operation         cost [MEUR]')
mDEPLOhy.vPeriodPurchaseCost   = Var(mDEPLOhy.p,      within=NonNegativeReals,                                                                                                                doc='period power purchase                 cost [MEUR]')
mDEPLOhy.vPeriodSaleRevenue    = Var(mDEPLOhy.p,      within=NonNegativeReals,                                                                                                                doc='period power sell                  revenue [MEUR]')
mDEPLOhy.vPeriodResRevenue     = Var(mDEPLOhy.p,      within=NonNegativeReals,                                                                                                                doc='period balancing reserves          revenue [MEUR]')
mDEPLOhy.vTotalSCost           = Var(                 within=           Reals,                                                                                                                doc='total  system                         cost [MEUR]')
mDEPLOhy.vTotalFCost           = Var(                 within=NonNegativeReals,                                                                                                                doc='total  system fixed capital           cost [MEUR]')
mDEPLOhy.vTotalOCost           = Var(                 within=NonNegativeReals,                                                                                                                doc='total  system fixed operation         cost [MEUR]')
mDEPLOhy.vTotalMCost           = Var(                 within=NonNegativeReals,                                                                                                                doc='total  system market                  cost [MEUR]')
mDEPLOhy.vTotalMRevenue        = Var(                 within=NonNegativeReals,                                                                                                                doc='total  system market               revenue [MEUR]')
mDEPLOhy.vPeriodOutflows       = Var(mDEPLOhy.ps,     within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc      :(mDEPLOhy.pMinPeriodOutflows[p   ],mDEPLOhy.pMaxPeriodOutflows[p         ]), doc='period           hydrogen outflows          [tH2]')
mDEPLOhy.vTotalOutflows        = Var(                 within=NonNegativeReals, bounds=lambda mDEPLOhy           :(mDEPLOhy.pMinTotalOutflows       ,mDEPLOhy.pMaxTotalOutflows             ), doc='total            hydrogen outflows          [tH2]')
mDEPLOhy.vDiscTotalOutflows    = Var(                 within=NonNegativeReals, bounds=lambda mDEPLOhy           :(mDEPLOhy.pMinDiscTotalOutflows   ,mDEPLOhy.pMaxDiscTotalOutflows         ), doc='total discounted hydrogen outflows          [tH2]')
mDEPLOhy.vOutput               = Var(mDEPLOhy.psng ,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,g  :(0.0,                              mDEPLOhy.pMaxPower         [p,sc,n,g  ]), doc='output of a      unit                        [GW]')
mDEPLOhy.vOutput2ndBlock       = Var(mDEPLOhy.psnsu,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,su :(0.0,                              mDEPLOhy.pMaxPower         [p,sc,n,su ]), doc='output of a BESS unit   above min value      [GW]')
mDEPLOhy.vESSCharge            = Var(mDEPLOhy.psnes,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,es :(0.0,                              mDEPLOhy.pMaxCharge        [p,sc,n,es ]), doc='ESS    charge power                          [GW]')
mDEPLOhy.vESSCharge2ndBlock    = Var(mDEPLOhy.psnnr,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,nr :(0.0,                              mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,nr ]), doc='ESS    charge power above min value          [GW]')
mDEPLOhy.vReserveUpC           = Var(mDEPLOhy.psnnr,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,nr :(0.0,                              mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,nr ]), doc='upward   balancing reserve when    charging  [GW]')
mDEPLOhy.vReserveDwC           = Var(mDEPLOhy.psnnr,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,nr :(0.0,                              mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,nr ]), doc='downward balancing reserve when    charging  [GW]')
mDEPLOhy.vReserveUpD           = Var(mDEPLOhy.psnsu,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,su :(0.0,                              mDEPLOhy.pMaxPower         [p,sc,n,su ]), doc='upward   balancing reserve when discharging  [GW]')
mDEPLOhy.vReserveDwD           = Var(mDEPLOhy.psnsu,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,su :(0.0,                              mDEPLOhy.pMaxPower         [p,sc,n,su ]), doc='downward balancing reserve when discharging  [GW]')
mDEPLOhy.vCurtailment          = Var(mDEPLOhy.psnppa, within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,ppa:(0.0,                              mDEPLOhy.pMaxPower         [p,sc,n,ppa]), doc='ppa curtailment                              [GW]')
mDEPLOhy.vH2Production         = Var(mDEPLOhy.psnel,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,el :(0.0,                              mDEPLOhy.pMaxH2Prod        [p         ]), doc='hydrogen     production                   [tH2/h]')
mDEPLOhy.vH2Prod2ndBlock       = Var(mDEPLOhy.psnel,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,el :(0.0,                              mDEPLOhy.pMaxH2Prod2ndBlock[p         ]), doc='hydrogen     production above min value   [tH2/h]')
mDEPLOhy.vH2Outflows           = Var(mDEPLOhy.psnhs,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,hs :(mDEPLOhy.pMinOutflows [p,sc,n,hs],mDEPLOhy.pMaxOutflows      [p,sc,n,hs ]), doc='hydrogen outflows                         [tH2/h]')
mDEPLOhy.vESSInventory         = Var(mDEPLOhy.psnsu,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,su :(0.0                              ,mDEPLOhy.pMaxStorage       [p,sc,n,su ]), doc='BESS      inventory                         [GWh]')
mDEPLOhy.vESSIniInventory      = Var(mDEPLOhy.pssu ,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,  su :(0.0                              ,mDEPLOhy.pRatedMaxStorage  [       su ]), doc='initial inventory for BESS candidates       [GWh]')
mDEPLOhy.vESSSpillage          = Var(mDEPLOhy.psnsu,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,su :(0.0                              ,mDEPLOhy.pMaxStorage       [p,sc,n,su ]), doc='ESS spillage                                [GWh]')
mDEPLOhy.vH2Inventory          = Var(mDEPLOhy.psnhs,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,hs :(0.0                              ,mDEPLOhy.pH2MaxStorage     [p,sc,n,hs ]), doc='H2Storage inventory                         [tH2]')
mDEPLOhy.vH2IniInventory       = Var(mDEPLOhy.pshs ,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,  hs :(0.0                              ,mDEPLOhy.pH2RatedMaxStorage[       hs ]), doc='initial inventory for H2Storage candidates  [tH2]')
mDEPLOhy.vH2Spillage           = Var(mDEPLOhy.psnhs,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,hs :(0.0                              ,mDEPLOhy.pH2MaxStorage     [p,sc,n,hs ]), doc='H2  spillage                                [tH2]')
mDEPLOhy.vFlow                 = Var(mDEPLOhy.psnla,  within=NonNegativeReals,                                                                                                                doc='electricity flow                             [GW]')
mDEPLOhy.vCommInvestment       = Var(mDEPLOhy.psnel,  within=UnitInterval,                                                                                                                    doc='EZ investment bounded by commitment        [p.u.]')
mDEPLOhy.vSUInvestment         = Var(mDEPLOhy.psnel,  within=UnitInterval,                                                                                                                    doc='EZ investment bounded by start-up          [p.u.]')
mDEPLOhy.vSDInvestment         = Var(mDEPLOhy.psnel,  within=UnitInterval,                                                                                                                    doc='EZ investment bounded by shutdown          [p.u.]')
mDEPLOhy.vSBInvestment         = Var(mDEPLOhy.psnel,  within=UnitInterval,                                                                                                                    doc='EZ investment bounded by standby           [p.u.]')
mDEPLOhy.vPurchase             = Var(mDEPLOhy.psnnd,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,nd :(0.0,                              mDEPLOhy.pGCPCapacity                  ), doc='power purchase in node                       [GW]')
mDEPLOhy.vSale                 = Var(mDEPLOhy.psnnd,  within=NonNegativeReals, bounds=lambda mDEPLOhy,p,sc,n,nd :(0.0,                              mDEPLOhy.pGCPCapacity                  ), doc='power sale     in node                       [GW]')
mDEPLOhy.vContractedPower      = Var(mDEPLOhy.pm,     within=NonNegativeReals, bounds=lambda mDEPLOhy,p,m       :(0.0,                              mDEPLOhy.pGCPCapacity                  ), doc='contracted power per market period           [GW]')

if mDEPLOhy.pIndBinInvest() == 0:
    mDEPLOhy.vGlobalInvestment = Var(                 within=UnitInterval,                                                                                                                    doc='EZ   investment decision unique along years [0,1]')
    mDEPLOhy.vUnitInvestment   = Var(mDEPLOhy.pgc,    within=UnitInterval,                                                                                                                    doc='unit investment decision exists in a year   [0,1]')
    mDEPLOhy.vUnitInvestPer    = Var(mDEPLOhy.pgc,    within=UnitInterval,                                                                                                                    doc='unit investment decision done   in a year   [0,1]')
    mDEPLOhy.vPPAInvestment    = Var(mDEPLOhy.pppa,   within=UnitInterval,                                                                                                                    doc='PPA  investment decision exists in a year   [0,1]')
    mDEPLOhy.vPPAInvestPer     = Var(mDEPLOhy.pppa,   within=UnitInterval,                                                                                                                    doc='PPA  investment decision done   in a year   [0,1]')
else:
    mDEPLOhy.vGlobalInvestment = Var(                 within=Binary,                                                                                                                          doc='EZ   investment decision unique along years {0,1}')
    mDEPLOhy.vUnitInvestment   = Var(mDEPLOhy.pgc,    within=Binary,                                                                                                                          doc='unit investment decision exists in a year   {0,1}')
    mDEPLOhy.vUnitInvestPer    = Var(mDEPLOhy.pgc,    within=Binary,                                                                                                                          doc='unit investment decision done   in a year   {0,1}')
    mDEPLOhy.vPPAInvestment    = Var(mDEPLOhy.pppa,   within=Binary,                                                                                                                          doc='PPA  investment decision exists in a year   {0,1}')
    mDEPLOhy.vPPAInvestPer     = Var(mDEPLOhy.pppa,   within=Binary,                                                                                                                          doc='PPA  investment decision exists in a year   {0,1}')

if mDEPLOhy.pIndBinRetire() == 0:
    mDEPLOhy.vUnitRetirement   = Var(mDEPLOhy.pgd,    within=UnitInterval,                                                                                                                    doc='unit retirement decision exists in a year   [0,1]')
    mDEPLOhy.vUnitRetirePer    = Var(mDEPLOhy.pgd,    within=UnitInterval,                                                                                                                    doc='unit retirement decision done   in a year   [0,1]')
else:
    mDEPLOhy.vUnitRetirement   = Var(mDEPLOhy.pgd,    within=Binary,                                                                                                                          doc='unit retirement decision exists in a year   {0,1}')
    mDEPLOhy.vUnitRetirePer    = Var(mDEPLOhy.pgd,    within=Binary,                                                                                                                          doc='unit retirement decision done   in a year   {0,1}')

if mDEPLOhy.pIndBinOperation() == 0:
    mDEPLOhy.vCommitment       = Var(mDEPLOhy.psnel,  within=UnitInterval,     initialize=0.0,                                                                                                doc='EZ commitment state                         (0,1)')
    mDEPLOhy.vStandBy          = Var(mDEPLOhy.psnel,  within=UnitInterval,     initialize=0.0,                                                                                                doc='EZ standby    state                         (0,1)')
    mDEPLOhy.vOff              = Var(mDEPLOhy.psnel,  within=UnitInterval,     initialize=0.0,                                                                                                doc='EZ idle       state                         (0,1)')
    mDEPLOhy.vStartUp          = Var(mDEPLOhy.psnel,  within=UnitInterval,     initialize=0.0,                                                                                                doc='EZ start-up                                 (0,1)')
    mDEPLOhy.vShutDown         = Var(mDEPLOhy.psnel,  within=UnitInterval,     initialize=0.0,                                                                                                doc='EZ shutdown                                 (0,1)')
    mDEPLOhy.vChargeInd        = Var(mDEPLOhy.psnsu,  within=UnitInterval,     initialize=0.0,                                                                                                doc='BESS unit charge indicator                  (0,1)')
    mDEPLOhy.vActivationC      = Var(mDEPLOhy.psnnr,  within=UnitInterval,     initialize=0.0,                                                                                                doc='upward activation when    charging          (0,1)')
    mDEPLOhy.vActivationD      = Var(mDEPLOhy.psnsu,  within=UnitInterval,     initialize=0.0,                                                                                                doc='upward activation when discharging          (0,1)')
else:
    mDEPLOhy.vCommitment       = Var(mDEPLOhy.psnel,  within=Binary,           initialize=0  ,                                                                                                doc='EZ commitment state                         {0,1}')
    mDEPLOhy.vStandBy          = Var(mDEPLOhy.psnel,  within=Binary,           initialize=0  ,                                                                                                doc='EZ standby    state                         {0,1}')
    mDEPLOhy.vOff              = Var(mDEPLOhy.psnel,  within=Binary,           initialize=0  ,                                                                                                doc='EZ idle       state                         {0,1}')
    mDEPLOhy.vStartUp          = Var(mDEPLOhy.psnel,  within=Binary,           initialize=0  ,                                                                                                doc='EZ start-up                                 {0,1}')
    mDEPLOhy.vShutDown         = Var(mDEPLOhy.psnel,  within=Binary,           initialize=0  ,                                                                                                doc='EZ shutdown                                 {0,1}')
    mDEPLOhy.vChargeInd        = Var(mDEPLOhy.psnsu,  within=Binary,           initialize=0  ,                                                                                                doc='BESS unit charge indicator                  {0,1}')
    mDEPLOhy.vActivationC      = Var(mDEPLOhy.psnnr,  within=Binary,           initialize=0  ,                                                                                                doc='upward activation when    charging          {0,1}')
    mDEPLOhy.vActivationD      = Var(mDEPLOhy.psnsu,  within=Binary,           initialize=0  ,                                                                                                doc='upward activation when discharging          {0,1}')


nFixedVariables = 0.0

# Fix period hydrogen outflows
for p,sc in mDEPLOhy.ps:
    if pAnnualOutflows >= mDEPLOhy.pMinAnnualOutflows[p]:
        mDEPLOhy.vPeriodOutflows[p,sc].fix( pAnnualOutflows                * mDEPLOhy.pPeriodWeight  [p])
        nFixedVariables += 1
    else:
        mDEPLOhy.vPeriodOutflows[p,sc].fix( mDEPLOhy.pMinAnnualOutflows[p] * mDEPLOhy.pPeriodWeight  [p])
        nFixedVariables += 1

# Fix nodes where the system cannot purchase or sell power
for p,sc,n,nd in mDEPLOhy.psnnd:
    if nd not in mDEPLOhy.ndb:
        mDEPLOhy.vPurchase [p,sc,n,nd].fix(0.0)
        nFixedVariables += 1
    if nd not in mDEPLOhy.nds:
        mDEPLOhy.vSale     [p,sc,n,nd].fix(0.0)
        nFixedVariables += 1

# Set upper bound for purchases based on certifications' conditions
for p,sc,n,ndb in mDEPLOhy.psnnb:
    if mDEPLOhy.pIndLCF() == 1 and mDEPLOhy.pIndBinInvest() == 2:
        mDEPLOhy.vPurchase[p,sc,n,ndb].setub(mDEPLOhy.pMaxGCP[p,sc,n,ndb] + mDEPLOhy.pMaxGCPzero[p,sc,n,ndb])
    if mDEPLOhy.pIndRFNBO() == 1:
        if mDEPLOhy.pAPRE[p] <= 0.9:
            mDEPLOhy.vPurchase[p,sc,n,ndb].fix(0.0)
            nFixedVariables += 1
        else:
            mDEPLOhy.vPurchase[p,sc,n,ndb].setub(mDEPLOhy.pGCPCapacity)

# Fix contracted power values if indicated
for p,m in mDEPLOhy.pm:
    if mDEPLOhy.pFixedPower[p,m] > 0.0:
        mDEPLOhy.vContractedPower[p,m].fix(mDEPLOhy.pFixedPower[p,m])
        nFixedVariables += 1

# Relax binary condition in unit investment decisions
for p,gc in mDEPLOhy.pgc:
    if mDEPLOhy.pIndBinInvest() != 0 and mDEPLOhy.pIndBinUnitInvest[gc ] == 0:
        mDEPLOhy.vUnitInvestment[p,gc ].domain = UnitInterval
    if mDEPLOhy.pIndBinInvest() == 0 and mDEPLOhy.pIndBinUnitInvest[gc ] == 1:
        mDEPLOhy.vUnitInvestment[p,gc ].domain = Binary
    if mDEPLOhy.pIndBinInvest() == 1:
        mDEPLOhy.vUnitInvestment[p,gc ].fix(1)
        mDEPLOhy.vUnitInvestPer [p,gc ].fix(0)
        nFixedVariables += 2
    if mDEPLOhy.pIndBinInvest() == 2:
        mDEPLOhy.vUnitInvestment[p,gc ].fix(0)
        mDEPLOhy.vUnitInvestPer [p,gc ].fix(0)
        nFixedVariables += 2

for p,ppa in mDEPLOhy.pppa:
    if mDEPLOhy.pIndBinInvest() != 0 and mDEPLOhy.pIndBinUnitInvest[ppa] == 0:
        mDEPLOhy.vPPAInvestment [p,ppa].domain = UnitInterval
    if mDEPLOhy.pIndBinInvest() == 0 and mDEPLOhy.pIndBinUnitInvest[ppa] == 1:
        mDEPLOhy.vPPAInvestment [p,ppa].domain = Binary
    if mDEPLOhy.pIndBinInvest() == 1:
        mDEPLOhy.vPPAInvestment [p,ppa].fix(1)
        mDEPLOhy.vPPAInvestPer  [p,ppa].fix(0)
        nFixedVariables += 2
    if mDEPLOhy.pIndBinInvest() == 2:
        mDEPLOhy.vPPAInvestment [p,ppa].fix(0)
        mDEPLOhy.vPPAInvestPer  [p,ppa].fix(0)
        nFixedVariables += 2

if mDEPLOhy.pIndBinInvest() != 0:
    mDEPLOhy.vGlobalInvestment.fix(1)
    nFixedVariables += 1

# Relax binary condition in unit retirement decisions
for p,gd in mDEPLOhy.pgd:
    if mDEPLOhy.pIndBinRetire() != 0 and mDEPLOhy.pIndBinUnitRetire[gd] == 0:
        mDEPLOhy.vUnitRetirement[p,gd].domain = UnitInterval
    if mDEPLOhy.pIndBinRetire() == 0 and mDEPLOhy.pIndBinUnitRetire[gd] == 1:
        mDEPLOhy.vUnitRetirement[p,gd].domain = Binary
    if mDEPLOhy.pIndBinRetire() == 2:
        mDEPLOhy.vUnitRetirement[p,gd].fix(0)
        mDEPLOhy.vUnitRetirePer [p,gd].fix(0)
        nFixedVariables += 2

# Fixing retirement variables
for p,gd in mDEPLOhy.pgd:
    if                                         mDEPLOhy.pFinalPeriod[gd] >  p:
        mDEPLOhy.vUnitRetirement[p,gd].fix(0)
        nFixedVariables += 1
    if mDEPLOhy.pIndBinUnitRetire[gd] == 1 and mDEPLOhy.pFinalPeriod[gd] <= p:
        mDEPLOhy.vUnitRetirement[p,gd].fix(1)
        nFixedVariables += 1

# Incoming and outgoing lines (lin) (lout)
lin   = defaultdict(list)
lout  = defaultdict(list)
for ni,nf,cc in mDEPLOhy.la:
    lin[nf].append((ni,cc))
    lout[ni].append((nf,cc))

# Nodes to all units (g2n)
g2n = defaultdict(list)
for nd,g  in mDEPLOhy.n2g:
    g2n[nd].append(g)

# Nodes to es units (e2n)
e2n = defaultdict(list)
for nd,es in mDEPLOhy.nd*mDEPLOhy.es:
    if (nd,es) in mDEPLOhy.n2g:
        e2n[nd].append(es)

# Nodes to electrolyzers (h2n)
h2n = defaultdict(list)
for nd,el in mDEPLOhy.nd*mDEPLOhy.el:
    if (nd,el) in mDEPLOhy.n2g:
        h2n[nd].append(el)

# Nodes to batteries (s2n)
s2n = defaultdict(list)
for nd,su in mDEPLOhy.nd*mDEPLOhy.su:
    if (nd,su) in mDEPLOhy.n2g:
        s2n[nd].append(su)

# Nodes to PPA (p2n)
p2n = defaultdict(list)
for nd,ppa in mDEPLOhy.nd*mDEPLOhy.ppa:
    if (nd,ppa) in mDEPLOhy.n2g:
        p2n[nd].append(ppa)

for p,sc,n,g  in mDEPLOhy.psng :
    # if there is no max power there is no output
    if mDEPLOhy.pMaxPower          [p,sc,n,g ] ==  0.0:
        mDEPLOhy.vOutput           [p,sc,n,g ].fix(0.0)
        nFixedVariables += 1

for p,sc,n,es in mDEPLOhy.psnes:
    # ESS with no charge capacity or not storage capacity can't charge
    if mDEPLOhy.pMaxCharge         [p,sc,n,es] ==  0.0:
        mDEPLOhy.vESSCharge        [p,sc,n,es].fix(0.0)
        nFixedVariables += 1

for p,sc,n,su in mDEPLOhy.psnsu:
    # BESS with no discharge capacity or not storage capacity can't charge/store
    if mDEPLOhy.pMaxPower          [p,sc,n,su] ==  0.0:
        mDEPLOhy.vOutput2ndBlock   [p,sc,n,su].fix(0.0)
        nFixedVariables += 1
    if mDEPLOhy.pMaxCharge         [p,sc,n,su] ==  0.0:
        mDEPLOhy.vESSCharge2ndBlock[p,sc,n,su].fix(0.0)
        nFixedVariables += 1
    if mDEPLOhy.pMaxStorage        [p,sc,n,su] ==  0.0:
        mDEPLOhy.vESSInventory     [p,sc,n,su].fix(0.0)
        mDEPLOhy.vESSSpillage      [p,sc,n,su].fix(0.0)
        nFixedVariables += 2

for p,sc,n,hs in mDEPLOhy.psnhs:
    # H2Storage with no capacity can't store
    if mDEPLOhy.pH2MaxStorage      [p,sc,n,hs] ==  0.0:
        mDEPLOhy.vH2Inventory      [p,sc,n,hs].fix(0.0)
        nFixedVariables += 1
    # H2Storage with no outflows can't deliver
    if mDEPLOhy.pMaxOutflows       [p,sc,n,hs] ==  0.0:
        mDEPLOhy.vH2Outflows       [p,sc,n,hs].fix(0.0)
        nFixedVariables += 1

for p,sc,su in mDEPLOhy.pssu:
    if su not in mDEPLOhy.ec:
        mDEPLOhy.vESSIniInventory [p,sc,su].fix(0.0)
        nFixedVariables += 1
    if mDEPLOhy.pRatedMaxStorage  [     su] ==  0.0:
        mDEPLOhy.vESSIniInventory [p,sc,su].fix(0.0)
        nFixedVariables += 1

for p,sc,hs in mDEPLOhy.pshs:
    if hs not in mDEPLOhy.hc:
        mDEPLOhy.vH2IniInventory  [p,sc,  hs].fix(0.0)
        nFixedVariables += 1
    if mDEPLOhy.pH2RatedMaxStorage[       hs] ==  0.0:
        mDEPLOhy.vH2IniInventory  [p,sc,  hs].fix(0.0)
        nFixedVariables += 1

# Fix variables if no reserves are considered
if mDEPLOhy.pIndReserves() == 0:
    for p         in mDEPLOhy.p:
        mDEPLOhy.vPeriodResRevenue[p        ].fix(0.0)
        nFixedVariables += 1
    for p,sc,n,su in mDEPLOhy.psnsu:
        mDEPLOhy.vReserveUpD      [p,sc,n,su].fix(0.0)
        mDEPLOhy.vReserveDwD      [p,sc,n,su].fix(0.0)
        nFixedVariables += 2
        if mDEPLOhy.pIndBinOperation() == 0:
            mDEPLOhy.vActivationD [p,sc,n,su].fix(0.0)
            nFixedVariables += 1
        else:
            mDEPLOhy.vActivationD [p,sc,n,su].fix(0  )
            nFixedVariables += 1
    for p,sc,n,nr in mDEPLOhy.psnnr:
        mDEPLOhy.vReserveUpC      [p,sc,n,nr].fix(0.0)
        mDEPLOhy.vReserveDwC      [p,sc,n,nr].fix(0.0)
        nFixedVariables += 2
        if mDEPLOhy.pIndBinOperation() == 0:
            mDEPLOhy.vActivationC [p,sc,n,nr].fix(0.0)
            nFixedVariables += 1
        else:
            mDEPLOhy.vActivationC [p,sc,n,nr].fix(0  )
            nFixedVariables += 1

for p,sc,n,ppa in mDEPLOhy.psnppa:
    if mDEPLOhy.pMaxPower[p,sc,n,ppa] == 0.0:
        mDEPLOhy.vCurtailment    [p,sc,n,ppa].fix(0.0)
        nFixedVariables += 1

# Do not install/retire power plants if not allowed in this period
for p,gc in mDEPLOhy.pgc:
    if mDEPLOhy.pInitialPeriod[gc] > p:
        mDEPLOhy.vUnitInvestment[p,gc].fix(0)
        mDEPLOhy.vUnitInvestPer [p,gc].fix(0)
        nFixedVariables += 2
    if mDEPLOhy.pFinalPeriod[gc] < p:
        mDEPLOhy.vUnitInvestPer [p,gc].fix(0)
        nFixedVariables += 1
    if mDEPLOhy.pInitialPeriod[gc] == mDEPLOhy.p.first() and mDEPLOhy.pFinalPeriod[gc] >= mDEPLOhy.p.last():
        mDEPLOhy.vUnitInvestPer[p,gc].fix(0)
        nFixedVariables += 1
    if mDEPLOhy.pInitialPeriod[gc] < p <= mDEPLOhy.pFinalPeriod[gc]:
        mDEPLOhy.vUnitInvestPer[p,gc].fix(0)
        nFixedVariables += 1

for p,ppa in mDEPLOhy.pppa:
    if mDEPLOhy.pInitialPeriod[ppa] > p:
        mDEPLOhy.vPPAInvestment[p,ppa].fix(0)
        mDEPLOhy.vPPAInvestPer [p,ppa].fix(0)
        nFixedVariables += 2
    if mDEPLOhy.pFinalPeriod[ppa] < p:
        mDEPLOhy.vPPAInvestPer [p,ppa].fix(0)
        nFixedVariables += 1
    if mDEPLOhy.pInitialPeriod[ppa] == mDEPLOhy.p.first() and mDEPLOhy.pFinalPeriod[ppa] > mDEPLOhy.p.last():
        mDEPLOhy.vPPAInvestPer [p,ppa].fix(0)
        nFixedVariables += 1
    if mDEPLOhy.pInitialPeriod[ppa] < p < mDEPLOhy.pFinalPeriod[ppa]:
        mDEPLOhy.vPPAInvestPer [p,ppa].fix(0)
        nFixedVariables += 1


for p,gd in mDEPLOhy.pgd:
    if mDEPLOhy.pInitialPeriod[gd] > p:
        mDEPLOhy.vUnitRetirement[p,gd].fix(0)
        mDEPLOhy.vUnitRetirePer [p,gd].fix(0)
        nFixedVariables += 2
    if mDEPLOhy.pFinalPeriod[gd] < p:
        mDEPLOhy.vUnitRetirePer [p,gd].fix(0)
        nFixedVariables += 1

# Remove power plants and lines not installed in this period

for p,sc,n,g   in mDEPLOhy.psng:
    if mDEPLOhy.pInitialPeriod[g  ] > p or mDEPLOhy.pFinalPeriod[g  ] <= p:
        mDEPLOhy.vOutput           [p,sc,n,g  ].fix(0.0)
        nFixedVariables += 1

for p,g in mDEPLOhy.pg:
    if mDEPLOhy.pInitialPeriod[g  ] > p or mDEPLOhy.pFinalPeriod[g  ] <= p:
        mDEPLOhy.vUnitOpCost       [p,     g  ].fix(0.0)
        nFixedVariables += 1

for p,sc,n,ppa in mDEPLOhy.psnppa:
    if mDEPLOhy.pInitialPeriod[ppa] > p or mDEPLOhy.pFinalPeriod[ppa] <= p:
        mDEPLOhy.vCurtailment      [p,sc,n,ppa].fix(0.0)
        nFixedVariables += 1

for p,sc,n,es in mDEPLOhy.psnes:
    if mDEPLOhy.pInitialPeriod[es ] > p or mDEPLOhy.pFinalPeriod[es ] <= p:
        mDEPLOhy.vESSCharge        [p,sc,n,es].fix(0.0)
        mDEPLOhy.vOutput           [p,sc,n,es].fix(0.0)
        nFixedVariables += 2

for p,sc,n,hs in mDEPLOhy.psnhs:
    if mDEPLOhy.pInitialPeriod[hs] > p or mDEPLOhy.pFinalPeriod[hs] <= p:
        mDEPLOhy.vH2Outflows       [p,sc,n,hs].fix(0.0)
        mDEPLOhy.vH2Inventory      [p,sc,n,hs].fix(0.0)
        mDEPLOhy.vH2Spillage       [p,sc,n,hs].fix(0.0)
        nFixedVariables += 3

for p,sc,hs in mDEPLOhy.pshs:
    if mDEPLOhy.pInitialPeriod[hs] > p or mDEPLOhy.pFinalPeriod[hs] <= p:
        mDEPLOhy.vH2IniInventory   [p,sc,  hs].fix(0.0)
        nFixedVariables += 1

for p,sc,n,el in mDEPLOhy.psnel:
    if mDEPLOhy.pInitialPeriod[el] > p or mDEPLOhy.pFinalPeriod[el] <= p:
        mDEPLOhy.vH2Production     [p,sc,n,el].fix(0.0)
        mDEPLOhy.vH2Prod2ndBlock   [p,sc,n,el].fix(0.0)
        mDEPLOhy.vCommInvestment   [p,sc,n,el].fix(0.0)
        mDEPLOhy.vSUInvestment     [p,sc,n,el].fix(0.0)
        mDEPLOhy.vSDInvestment     [p,sc,n,el].fix(0.0)
        mDEPLOhy.vSBInvestment     [p,sc,n,el].fix(0.0)
        nFixedVariables += 6
        if mDEPLOhy.pIndBinOperation() == 0:
            mDEPLOhy.vCommitment   [p,sc,n,el].fix(0.0)
            mDEPLOhy.vStandBy      [p,sc,n,el].fix(0.0)
            mDEPLOhy.vOff          [p,sc,n,el].fix(0.0)
            mDEPLOhy.vStartUp      [p,sc,n,el].fix(0.0)
            mDEPLOhy.vShutDown     [p,sc,n,el].fix(0.0)
            nFixedVariables += 5
        else:
            mDEPLOhy.vCommitment   [p,sc,n,el].fix(0  )
            mDEPLOhy.vStandBy      [p,sc,n,el].fix(0  )
            mDEPLOhy.vOff          [p,sc,n,el].fix(0  )
            mDEPLOhy.vStartUp      [p,sc,n,el].fix(0  )
            mDEPLOhy.vShutDown     [p,sc,n,el].fix(0  )
            nFixedVariables += 5
    if mDEPLOhy.pIndPiecewise() != 1:
        mDEPLOhy.vH2Prod2ndBlock   [p,sc,n,el].fix(0.0)
        nFixedVariables += 1

for p,sc,n,el in mDEPLOhy.psnel:
    if mDEPLOhy.pIndTwoStates() == 1:
        mDEPLOhy.vSUInvestment     [p,sc,n,el].fix(0.0)
        mDEPLOhy.vSDInvestment     [p,sc,n,el].fix(0.0)
        nFixedVariables += 2
        if mDEPLOhy.pIndBinOperation() == 0:
            mDEPLOhy.vOff          [p,sc,n,el].fix(0.0)
            mDEPLOhy.vStartUp      [p,sc,n,el].fix(0.0)
            mDEPLOhy.vShutDown     [p,sc,n,el].fix(0.0)
            nFixedVariables += 3
        else:
            mDEPLOhy.vOff          [p,sc,n,el].fix(0  )
            mDEPLOhy.vStartUp      [p,sc,n,el].fix(0  )
            mDEPLOhy.vShutDown     [p,sc,n,el].fix(0  )
            nFixedVariables += 3

for p,sc,n,su in mDEPLOhy.psnsu:
    if mDEPLOhy.pInitialPeriod[su] > p or mDEPLOhy.pFinalPeriod[su] <= p:
        mDEPLOhy.vOutput2ndBlock   [p,sc,n,su].fix(0.0)
        mDEPLOhy.vESSInventory     [p,sc,n,su].fix(0.0)
        mDEPLOhy.vESSSpillage      [p,sc,n,su].fix(0.0)
        mDEPLOhy.vReserveUpD       [p,sc,n,su].fix(0.0)
        mDEPLOhy.vReserveDwD       [p,sc,n,su].fix(0.0)
        nFixedVariables += 5
        if mDEPLOhy.pIndBinOperation() == 0:
            mDEPLOhy.vChargeInd    [p,sc,n,su].fix(0.0)
            mDEPLOhy.vActivationD  [p,sc,n,su].fix(0.0)
            nFixedVariables += 2
        else:
            mDEPLOhy.vChargeInd    [p,sc,n,su].fix(0  )
            mDEPLOhy.vActivationD  [p,sc,n,su].fix(0  )
            nFixedVariables += 2

for p,sc,su in mDEPLOhy.pssu:
    if mDEPLOhy.pInitialPeriod[su] > p or mDEPLOhy.pFinalPeriod[su] <= p:
        mDEPLOhy.vESSIniInventory  [p,sc,  su].fix(0.0)
        nFixedVariables += 1

for p,sc,n,nr in mDEPLOhy.psnnr:
    if mDEPLOhy.pInitialPeriod[nr] > p or mDEPLOhy.pFinalPeriod[nr] <= p:
        mDEPLOhy.vESSCharge2ndBlock[p,sc,n,nr].fix(0.0)
        mDEPLOhy.vReserveUpC       [p,sc,n,nr].fix(0.0)
        mDEPLOhy.vReserveDwC       [p,sc,n,nr].fix(0.0)
        nFixedVariables += 3
        if mDEPLOhy.pIndBinOperation() == 0:
            mDEPLOhy.vActivationC  [p,sc,n,nr].fix(0.0)
            nFixedVariables += 1
        else:
            mDEPLOhy.vActivationC  [p,sc,n,nr].fix(0  )
            nFixedVariables += 1

# Lower and upper bounds for investment and retirement decisions
pEpsilon = 1e-4
if mDEPLOhy.pIndBinInvest() == 0:
    for p,gc in mDEPLOhy.pgc:
        if  mDEPLOhy.pLowerInvest[  gc ]() <       pEpsilon:
            mDEPLOhy.pLowerInvest[  gc ]   = 0
        if  mDEPLOhy.pUpperInvest[  gc ]() <       pEpsilon:
            mDEPLOhy.pUpperInvest[  gc ]   = 0
        if  mDEPLOhy.pLowerInvest[  gc ]() > 1.0 - pEpsilon:
            mDEPLOhy.pLowerInvest[  gc ]   = 1
        if  mDEPLOhy.pUpperInvest[  gc ]() > 1.0 - pEpsilon:
            mDEPLOhy.pUpperInvest[  gc ]   = 1
        if  mDEPLOhy.pLowerInvest[  gc ]() >   mDEPLOhy.pUpperInvest[gc ]():
            mDEPLOhy.pLowerInvest[  gc ]   =   mDEPLOhy.pUpperInvest[gc ]()
    mDEPLOhy.vUnitInvestment     [p,gc ].setlb(mDEPLOhy.pLowerInvest[gc ]())
    mDEPLOhy.vUnitInvestment     [p,gc ].setub(mDEPLOhy.pUpperInvest[gc ]())

if mDEPLOhy.pIndBinInvest() == 0:
    for p,ppa in mDEPLOhy.pppa:
        if  mDEPLOhy.pLowerInvest[  ppa]() <       pEpsilon:
            mDEPLOhy.pLowerInvest[  ppa]   = 0
        if  mDEPLOhy.pUpperInvest[  ppa]() <       pEpsilon:
            mDEPLOhy.pUpperInvest[  ppa]   = 0
        if  mDEPLOhy.pLowerInvest[  ppa]() > 1.0 - pEpsilon:
            mDEPLOhy.pLowerInvest[  ppa]   = 1
        if  mDEPLOhy.pUpperInvest[  ppa]() > 1.0 - pEpsilon:
            mDEPLOhy.pUpperInvest[  ppa]   = 1
        if  mDEPLOhy.pLowerInvest[  ppa]() >   mDEPLOhy.pUpperInvest[ppa]():
            mDEPLOhy.pLowerInvest[  ppa]   =   mDEPLOhy.pUpperInvest[ppa]()
    mDEPLOhy.vPPAInvestment      [p,ppa].setlb(mDEPLOhy.pLowerInvest[ppa]())
    mDEPLOhy.vPPAInvestment      [p,ppa].setub(mDEPLOhy.pUpperInvest[ppa]())

if mDEPLOhy.pIndBinRetire() == 0:
    for p,gd in mDEPLOhy.pgd:
        if mDEPLOhy.pFinalPeriod[gd] <= p:
            if  mDEPLOhy.pLowerRetire[  gd ]() < pEpsilon:
                mDEPLOhy.pLowerRetire[  gd ]   = 0
            if  mDEPLOhy.pUpperRetire[  gd ]() <       pEpsilon:
                mDEPLOhy.pUpperRetire[  gd ]   = 0
            if  mDEPLOhy.pLowerRetire[  gd ]() > 1.0 - pEpsilon:
                mDEPLOhy.pLowerRetire[  gd ]   = 1
            if  mDEPLOhy.pUpperRetire[  gd ]() > 1.0 - pEpsilon:
                mDEPLOhy.pUpperRetire[  gd ]   = 1
            if  mDEPLOhy.pLowerRetire[  gd ]() >   mDEPLOhy.pUpperRetire[gd ]():
                mDEPLOhy.pLowerRetire[  gd ]   =   mDEPLOhy.pUpperRetire[gd ]()
        mDEPLOhy.vUnitRetirement     [p,gd ].setlb(mDEPLOhy.pLowerRetire[gd ]())
        mDEPLOhy.vUnitRetirement     [p,gd ].setub(mDEPLOhy.pUpperRetire[gd ]())

# Fixing binary investment and retirement decisions
for p,gc in mDEPLOhy.pgc:
    if mDEPLOhy.pUpperInvest[gc]() == mDEPLOhy.pLowerInvest[gc]() and p >= mDEPLOhy.pInitialPeriod[gc]:
        if mDEPLOhy.pIndBinUnitInvest[  gc] == 0:
            mDEPLOhy.vUnitInvestment [p,gc].fix(mDEPLOhy.pLowerInvest[gc])
            nFixedVariables += 1
        if mDEPLOhy.pIndBinUnitInvest[  gc] == 1:
            mDEPLOhy.vUnitInvestment [p,gc].fix(1  )
            nFixedVariables += 1

for p,ppa in mDEPLOhy.pppa:
    if mDEPLOhy.pUpperInvest[ppa]() == mDEPLOhy.pLowerInvest[ppa]() and mDEPLOhy.pInitialPeriod[ppa] <= p < mDEPLOhy.pFinalPeriod[ppa]:
        if mDEPLOhy.pIndBinUnitInvest[  ppa] == 0:
            mDEPLOhy.vPPAInvestment  [p,ppa].fix(mDEPLOhy.pLowerInvest[ppa])
            nFixedVariables += 1
        if mDEPLOhy.pIndBinUnitInvest[  ppa] == 1:
            mDEPLOhy.vPPAInvestment  [p,ppa].fix(1  )
            nFixedVariables += 1
        
for p,gd in mDEPLOhy.pgd:
    if mDEPLOhy.pUpperRetire[gd]() == mDEPLOhy.pLowerRetire[gd]() and mDEPLOhy.pFinalPeriod[gd] <= p:
        if mDEPLOhy.pIndBinUnitRetire[  gd] == 0:
            mDEPLOhy.vUnitRetirement [p,gd].fix(mDEPLOhy.pLowerRetire[gd])
            nFixedVariables += 1
        if mDEPLOhy.pIndBinUnitRetire[  gd] == 1:
            mDEPLOhy.vUnitRetirement [p,gd].fix(1  )
            nFixedVariables += 1

for p,sc,su in mDEPLOhy.pssu:
    # fixing the ESS inventory at the last load level for every period and scenario if between storage limits
    if su not in mDEPLOhy.ec:
        if mDEPLOhy.pMinStorage[p,sc,mDEPLOhy.n.last(),su] <= mDEPLOhy.pInitialInventory[su] <= mDEPLOhy.pMaxStorage[p,sc,mDEPLOhy.n.last(),su]:
            mDEPLOhy.vESSInventory[p,sc,mDEPLOhy.n.last(),su].fix(mDEPLOhy.pInitialInventory[su])
            nFixedVariables += 1

for p,sc,hs in mDEPLOhy.pshs:
    # fixing the H2 inventory at the last load level for every period and scenario if between storage limits
    if hs not in mDEPLOhy.hc:
        if mDEPLOhy.pH2MinStorage[p,sc,mDEPLOhy.n.last(),hs] <= mDEPLOhy.pH2InitialInventory[hs] <= mDEPLOhy.pH2MaxStorage[p,sc,mDEPLOhy.n.last(),hs]:
            mDEPLOhy.vH2Inventory[p,sc,mDEPLOhy.n.last(),hs].fix(mDEPLOhy.pH2InitialInventory[hs])
            nFixedVariables += 1

mDEPLOhy.nFixedVariables = Param(initialize=round(nFixedVariables), within=NonNegativeIntegers, doc='number of fixed variables')

SettingUpDataTime = time.time() - StartTime
StartTime         = time.time()
print('Setting up input data                          ... ', round(SettingUpDataTime), 's')


# =========================
# Objective function
# =========================

# Investment costs

def eUnitInvCost(mDEPLOhy,p,gc):
    if   gc in mDEPLOhy.rc:
        return mDEPLOhy.vUnitICost[p,gc] == mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pRatedMaxPower    [gc] * mDEPLOhy.pInvestCost[gc] * mDEPLOhy.vUnitInvestment[p,gc] * 1e3
    elif gc in mDEPLOhy.ec:
        return mDEPLOhy.vUnitICost[p,gc] == mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pRatedMaxCharge   [gc] * mDEPLOhy.pInvestCost[gc] * mDEPLOhy.vUnitInvestment[p,gc] * 1e3
    elif gc in mDEPLOhy.hc:
        return mDEPLOhy.vUnitICost[p,gc] == mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pH2RatedMaxStorage[gc] * mDEPLOhy.pInvestCost[gc] * mDEPLOhy.vUnitInvestment[p,gc]
    else:
        Constraint.Skip
mDEPLOhy.eUnitInvCost         = Constraint(mDEPLOhy.pgc,  rule=eUnitInvCost,        doc='candidate units investment cost [MEUR]')

def eUnitRetCost(mDEPLOhy,p,gd):
    if   gd in mDEPLOhy.rd:
        return mDEPLOhy.vUnitRCost[p,gd] == mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pRatedMaxPower    [gd] * mDEPLOhy.pRetireCost[gd] * mDEPLOhy.vUnitRetirement[p,gd] * 1e3
    elif gd in mDEPLOhy.ed:
        return mDEPLOhy.vUnitRCost[p,gd] == mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pRatedMaxCharge   [gd] * mDEPLOhy.pRetireCost[gd] * mDEPLOhy.vUnitRetirement[p,gd] * 1e3
    elif gd in mDEPLOhy.hd:
        return mDEPLOhy.vUnitRCost[p,gd] == mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pH2RatedMaxStorage[gd] * mDEPLOhy.pRetireCost[gd] * mDEPLOhy.vUnitRetirement[p,gd]
    else:
        Constraint.Skip
mDEPLOhy.eUnitRetCost         = Constraint(mDEPLOhy.pgd,  rule=eUnitRetCost,        doc='candidate units retirement cost [MEUR]')

def ePeriodFCost(mDEPLOhy,p):
    return mDEPLOhy.vPeriodFCost[p] == sum(mDEPLOhy.vUnitICost[p,gc] for gc in mDEPLOhy.gc) + sum(mDEPLOhy.vUnitRCost[p,gd] for gd in mDEPLOhy.gd)
mDEPLOhy.ePeriodFCost         = Constraint(mDEPLOhy.p,    rule=ePeriodFCost,        doc='period system fixed capital cost [MEUR]')

def eTotalFCost(mDEPLOhy):
    if len(mDEPLOhy.gc):
        return mDEPLOhy.vTotalFCost == sum(mDEPLOhy.vPeriodFCost[p] for p in mDEPLOhy.p)
    else:
        return Constraint.Skip
mDEPLOhy.eTotalFCost          = Constraint(               rule=eTotalFCost,         doc='total system fixed capital cost [MEUR]')


# Operation costs

def eCandidateOpCost(mDEPLOhy,p,gc):
    if mDEPLOhy.pInitialPeriod[gc] <= p < mDEPLOhy.pFinalPeriod[gc]:
        if   gc in mDEPLOhy.rc:
            return mDEPLOhy.vUnitOpCost[p,gc] ==      mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pOMCost  [gc  ] * mDEPLOhy.pRatedMaxPower        [gc] * 1e3 * mDEPLOhy.vUnitInvestment[p,gc]
        elif gc in mDEPLOhy.hc:
            return mDEPLOhy.vUnitOpCost[p,gc] ==      mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pOMCost  [gc  ] * mDEPLOhy.pH2RatedMaxStorage    [gc]       * mDEPLOhy.vUnitInvestment[p,gc]
        elif gc in mDEPLOhy.ec and gc not in mDEPLOhy.el:
            return mDEPLOhy.vUnitOpCost[p,gc] ==      mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pOMCost  [gc  ] * mDEPLOhy.pRatedMaxCharge       [gc] * 1e3 * mDEPLOhy.vUnitInvestment[p,gc]
        elif gc in mDEPLOhy.el:
            return mDEPLOhy.vUnitOpCost[p,gc] == (   (mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pOMCost  [gc  ] * mDEPLOhy.pRatedMaxCharge       [gc] * 1e3 * mDEPLOhy.vUnitInvestment[p,gc]) +
                                                  sum(mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pScenProb[p,sc] * 1e-3 * ((mDEPLOhy.pStartUpCost [gc] *    mDEPLOhy.vSUInvestment[p,sc,n,gc]) +
                                                                                                                       (mDEPLOhy.pShutDownCost[gc] *    mDEPLOhy.vSDInvestment[p,sc,n,gc]) ) for sc in PeriodScen[p] for n in mDEPLOhy.n))
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
mDEPLOhy.eCandidateOpCost     = Constraint(mDEPLOhy.pgc,  rule=eCandidateOpCost,    doc='candidate unit operation cost [MEUR]')

def eUnitOpCost(mDEPLOhy,p,g):
    if g not in mDEPLOhy.gc and mDEPLOhy.pInitialPeriod[g] <= p < mDEPLOhy.pFinalPeriod[g]:
        if   g  in mDEPLOhy.re:
            return mDEPLOhy.vUnitOpCost[p,g ] ==      mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pOMCost  [g   ] * mDEPLOhy.pRatedMaxPower        [g ] * 1e3
        elif g  in mDEPLOhy.hs:
            return mDEPLOhy.vUnitOpCost[p,g ] ==      mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pOMCost  [g   ] * mDEPLOhy.pH2RatedMaxStorage    [g ]
        elif g  in mDEPLOhy.es and g not in mDEPLOhy.el:
            return mDEPLOhy.vUnitOpCost[p,g ] ==      mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pOMCost  [g   ] * mDEPLOhy.pRatedMaxCharge       [g ] * 1e3
        elif g  in mDEPLOhy.el:
            return mDEPLOhy.vUnitOpCost[p,g ] == (   (mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pOMCost  [g   ] * mDEPLOhy.pRatedMaxCharge       [g ] * 1e3) +
                                                  sum(mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pScenProb[p,sc] * 1e-3 * ((mDEPLOhy.pStartUpCost [g ] *    mDEPLOhy.vSUInvestment[p,sc,n,g ]) +
                                                                                                                       (mDEPLOhy.pShutDownCost[g ] *    mDEPLOhy.vSDInvestment[p,sc,n,g ]) )    for sc in PeriodScen[p] for n in mDEPLOhy.n))
        elif g in mDEPLOhy.ppa:
            return mDEPLOhy.vUnitOpCost[p,g ] ==  sum(mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pScenProb[p,sc] * mDEPLOhy.pDuration[n] * mDEPLOhy.pPPACost[p,sc,n,g] * mDEPLOhy.vOutput[p,sc,n,g] for sc in PeriodScen[p] for n in mDEPLOhy.n)
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
mDEPLOhy.eUnitOpCost          = Constraint(mDEPLOhy.pg,   rule=eUnitOpCost,         doc='unit operation cost [MEUR]')

def ePeriodOpCost(mDEPLOhy,p):
    return mDEPLOhy.vPeriodOpCost[p]       ==  sum(mDEPLOhy.vUnitOpCost    [p,g] for g in mDEPLOhy.g)
mDEPLOhy.ePeriodOpCost        = Constraint(mDEPLOhy.p,    rule=ePeriodOpCost,       doc='period operation cost [MEUR]')

def ePeriodPurchaseCost(mDEPLOhy,p):
    return mDEPLOhy.vPeriodPurchaseCost[p] == ((sum(mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pScenProb[p,sc] * mDEPLOhy.pDuration[n] * mDEPLOhy.pEnergyCost[p,sc,n,ndb] * mDEPLOhy.vPurchase[p,sc,n,ndb] for sc in PeriodScen[p] for n in mDEPLOhy.n for ndb in mDEPLOhy.ndb) +
                                                sum(mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pPowerTariff[p,m] * mDEPLOhy.vContractedPower[p,m] for m in mDEPLOhy.m)) * (1 + mDEPLOhy.pEnergyTax)) * (1 + mDEPLOhy.pVAT)
mDEPLOhy.ePeriodPurchaseCost  = Constraint(mDEPLOhy.p,    rule=ePeriodPurchaseCost, doc='period power purchase cost [MEUR]')

def ePeriodSaleRevenue(mDEPLOhy,p):
    return mDEPLOhy.vPeriodSaleRevenue[p]  ==   sum(mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pScenProb[p,sc] * mDEPLOhy.pDuration[n] * mDEPLOhy.pSpotPrice [p,sc,n,nds]  * mDEPLOhy.vSale    [p,sc,n,nds] for sc in PeriodScen[p] for n in mDEPLOhy.n for nds in mDEPLOhy.nds)
mDEPLOhy.ePeriodSaleRevenue   = Constraint(mDEPLOhy.p,    rule=ePeriodSaleRevenue,  doc='period power sale revenue [MEUR]')

def ePeriodResRevenue(mDEPLOhy,p):
    return mDEPLOhy.vPeriodResRevenue[p]   ==  (sum(mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pScenProb[p,sc] *                         mDEPLOhy.pResUpPrice   [p,sc,n] * (sum( mDEPLOhy.vReserveUpC[p,sc,n,nr]                                   for nr in mDEPLOhy.nr) + sum( mDEPLOhy.vReserveUpD[p,sc,n,su]                                   for su in mDEPLOhy.su)) for sc in PeriodScen[p] for n in mDEPLOhy.n) +
                                                sum(mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pScenProb[p,sc] *                         mDEPLOhy.pResDwPrice   [p,sc,n] * (sum( mDEPLOhy.vReserveDwC[p,sc,n,nr]                                   for nr in mDEPLOhy.nr) + sum( mDEPLOhy.vReserveDwD[p,sc,n,su]                                   for su in mDEPLOhy.su)) for sc in PeriodScen[p] for n in mDEPLOhy.n) +
                                                sum(mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pScenProb[p,sc] * mDEPLOhy.pDuration[n] * mDEPLOhy.pEnergyUpPrice[p,sc,n] * (sum((mDEPLOhy.vReserveUpC[p,sc,n,nr] * mDEPLOhy.pActivationUp[p,sc,n]) for nr in mDEPLOhy.nr) + sum((mDEPLOhy.vReserveUpD[p,sc,n,su] * mDEPLOhy.pActivationUp[p,sc,n]) for su in mDEPLOhy.su)) for sc in PeriodScen[p] for n in mDEPLOhy.n) +
                                                sum(mDEPLOhy.pDiscountFactor[p] * mDEPLOhy.pScenProb[p,sc] * mDEPLOhy.pDuration[n] * mDEPLOhy.pEnergyDwPrice[p,sc,n] * (sum((mDEPLOhy.vReserveDwC[p,sc,n,nr] * mDEPLOhy.pActivationDw[p,sc,n]) for nr in mDEPLOhy.nr) + sum((mDEPLOhy.vReserveDwD[p,sc,n,su] * mDEPLOhy.pActivationDw[p,sc,n]) for su in mDEPLOhy.su)) for sc in PeriodScen[p] for n in mDEPLOhy.n) )
mDEPLOhy.ePeriodResRevenue    = Constraint(mDEPLOhy.p,    rule=ePeriodResRevenue,   doc='period reserves revenue [MEUR]')

def ePeriodSCost(mDEPLOhy,p):
    return mDEPLOhy.vPeriodSCost[p]        == mDEPLOhy.vPeriodFCost[p] + mDEPLOhy.vPeriodOpCost[p] + mDEPLOhy.vPeriodPurchaseCost[p] - mDEPLOhy.vPeriodSaleRevenue[p] - mDEPLOhy.vPeriodResRevenue[p]
mDEPLOhy.ePeriodSCost         = Constraint(mDEPLOhy.p,    rule=ePeriodSCost,        doc='period system cost [MEUR]')

def eTotalOCost(mDEPLOhy):
    return mDEPLOhy.vTotalOCost            ==  sum(mDEPLOhy.vPeriodOpCost[p]       for p   in mDEPLOhy.p )
mDEPLOhy.eTotalOCost          = Constraint(               rule=eTotalOCost,         doc='total system operation cost [MEUR]')

def eTotalMCost(mDEPLOhy):
    return mDEPLOhy.vTotalMCost            ==  sum(mDEPLOhy.vPeriodPurchaseCost[p] for p   in mDEPLOhy.p )
mDEPLOhy.eTotalMCost          = Constraint(               rule=eTotalMCost,         doc='total system market cost [MEUR]')

def eTotalMRevenue(mDEPLOhy):
    return mDEPLOhy.vTotalMRevenue         ==  sum((mDEPLOhy.vPeriodSaleRevenue[p] + mDEPLOhy.vPeriodResRevenue[p]) for p in mDEPLOhy.p)
mDEPLOhy.eTotalMRevenue       = Constraint(               rule=eTotalMRevenue,      doc='total system market revenue [MEUR]')

def eTotalSCost(mDEPLOhy):
    return mDEPLOhy.vTotalSCost            == mDEPLOhy.vTotalFCost + mDEPLOhy.vTotalOCost + mDEPLOhy.vTotalMCost - mDEPLOhy.vTotalMRevenue
mDEPLOhy.eTotalSCost          = Constraint(               rule=eTotalSCost,         doc='total system cost [MEUR]')


# Objective

def eObjective(mDEPLOhy):
    return mDEPLOhy.vTotalSCost
mDEPLOhy.eObjective            = Objective(rule=eObjective, sense=minimize,        doc='minimization of total system cost [MEUR]')

GeneratingOFTime = time.time() - StartTime
StartTime        = time.time()
print('Generating objective function                  ... ', round(GeneratingOFTime), 's')


# =========================
# Constraints
# =========================

# Investment & retirement constraints

def eEZInvestRule(mDEPLOhy,p,el):
    if el in mDEPLOhy.ec and p >= mDEPLOhy.pInitialPeriod[el]:
        return mDEPLOhy.vUnitInvestment[p,el] == mDEPLOhy.vGlobalInvestment
    else:
        return Constraint.Skip
mDEPLOhy.eEZInvestRule          = Constraint(mDEPLOhy.pel,    rule=eEZInvestRule,         doc='linking EZ investment decisions [p.u.]')

def eConsecutiveInvest(mDEPLOhy,p,gc):
    if p != mDEPLOhy.p.first():
        return mDEPLOhy.vUnitInvestment[mDEPLOhy.p.prev(p,1),gc ] + mDEPLOhy.vUnitInvestPer[p,gc ] == mDEPLOhy.vUnitInvestment[p,gc ]
    else:
        return Constraint.Skip
mDEPLOhy.eConsecutiveInvest     = Constraint(mDEPLOhy.pgc,    rule=eConsecutiveInvest,    doc='unit investment in consecutive periods [p.u.]')

def eConsecutivePPAInvest(mDEPLOhy,p,ppa):
    if p != mDEPLOhy.p.first() and p < mDEPLOhy.pFinalPeriod[ppa]:
        return mDEPLOhy.vPPAInvestment [mDEPLOhy.p.prev(p,1),ppa] + mDEPLOhy.vPPAInvestPer [p,ppa] == mDEPLOhy.vPPAInvestment [p,ppa]
    else:
        return Constraint.Skip
mDEPLOhy.eConsecutivePPAInvest  = Constraint(mDEPLOhy.pppa,   rule=eConsecutivePPAInvest, doc='PPA  investment in consecutive periods [p.u.]')

def eConsecutiveRetire(mDEPLOhy,p,gd):
    if p != mDEPLOhy.p.first():
        return mDEPLOhy.vUnitRetirement[mDEPLOhy.p.prev(p,1),gd ] + mDEPLOhy.vUnitRetirePer[p,gd ] == mDEPLOhy.vUnitRetirement[p,gd ]
    else:
        return Constraint.Skip
mDEPLOhy.eConsecutiveRetire     = Constraint(mDEPLOhy.pgd,    rule=eConsecutiveRetire,    doc='unit retirement in consecutive periods [p.u.]')

def eRetireDecision(mDEPLOhy,p,gd):
    if mDEPLOhy.pIndBinUnitRetire[gd] == 0 and p == mDEPLOhy.pFinalPeriod[gd]:
        return mDEPLOhy.vUnitRetirement[p,gd] == mDEPLOhy.vUnitInvestment[p,gd]
    else:
        return Constraint.Skip
mDEPLOhy.eRetireDecision        = Constraint(mDEPLOhy.pgd,    rule=eRetireDecision,       doc='retirement equal to investment decision [p.u.]')

def eInstallGenMax(mDEPLOhy,p,sc,n,gc):
    if mDEPLOhy.pMaxPower[p,sc,n,gc]:
        return mDEPLOhy.vOutput   [p,sc,n,gc ] / mDEPLOhy.pMaxPower [p,sc,n,gc ] <= mDEPLOhy.vUnitInvestment[p,gc]
    else:
        return Constraint.Skip
mDEPLOhy.eInstallGenMax         = Constraint(mDEPLOhy.psngc,  rule=eInstallGenMax,        doc='max output if installed gen unit [p.u.]')

def eInstallPPAMax(mDEPLOhy,p,sc,n,ppa):
    if mDEPLOhy.pIndBinInvest() != 2 and mDEPLOhy.pMaxPower[p,sc,n,ppa]:
        return mDEPLOhy.vOutput   [p,sc,n,ppa] / mDEPLOhy.pMaxPower [p,sc,n,ppa] == mDEPLOhy.vPPAInvestment[p,ppa]
    else:
        return Constraint.Skip
mDEPLOhy.eInstallPPAMax         = Constraint(mDEPLOhy.psnppa, rule=eInstallPPAMax,        doc='pay-as-produced PPA output [p.u.]')

def eInstallESSMax(mDEPLOhy,p,sc,n,ec):
    if mDEPLOhy.pMaxCharge[p,sc,n,ec]:
        return mDEPLOhy.vESSCharge[p,sc,n,ec ] / mDEPLOhy.pMaxCharge[p,sc,n,ec ] <= mDEPLOhy.vUnitInvestment[p,ec]
    else:
        return Constraint.Skip
mDEPLOhy.eInstallESSMax         = Constraint(mDEPLOhy.psnec,  rule=eInstallESSMax,        doc='max consumption if installed ESS unit [p.u.]')

def eUninstallComm(mDEPLOhy,p,sc,n,ed):
    if el in mDEPLOhy.ed:
        return mDEPLOhy.vCommitment[p,sc,n,ed] <= 1 - mDEPLOhy.vUnitRetirement[p,ed]
    else:
        return Constraint.Skip
mDEPLOhy.eUninstallComm         = Constraint(mDEPLOhy.psned,  rule=eUninstallComm,        doc='commitment if retired unit [p.u.]')

def eUninstallGen(mDEPLOhy,p,sc,n,gd):
    if mDEPLOhy.pMaxPower[p,sc,n,gd]:
        return mDEPLOhy.vOutput     [p,sc,n,gd] / mDEPLOhy.pMaxPower    [p,sc,n,gd ] <= 1 - mDEPLOhy.vUnitRetirement[p,gd]
    else:
        return Constraint.Skip
mDEPLOhy.eUninstallGen          = Constraint(mDEPLOhy.psngd,  rule=eUninstallGen,         doc='output if uninstalled gen unit [p.u.]')

def eUninstallESS(mDEPLOhy,p,sc,n,ed):
    if mDEPLOhy.pMaxCharge[p,sc,n,ed]:
        return mDEPLOhy.vESSCharge  [p,sc,n,ed] / mDEPLOhy.pMaxCharge   [p,sc,n,ed ] <= 1 - mDEPLOhy.vUnitRetirement[p,ed]
    else:
        return Constraint.Skip
mDEPLOhy.eUninstallESS          = Constraint(mDEPLOhy.psned,  rule=eUninstallESS,         doc='consumption if uninstalled ESS unit [p.u.]')

def eUninstallStorage(mDEPLOhy,p,sc,n,hd):
    if mDEPLOhy.pH2MaxStorage[p,sc,n,hd]:
        return mDEPLOhy.vH2Inventory[p,sc,n,hd] / mDEPLOhy.pH2MaxStorage[p,sc,n,hd ] <= 1 - mDEPLOhy.vUnitRetirement[p,hd]
    else:
        return Constraint.Skip
mDEPLOhy.eUninstallStorage      = Constraint(mDEPLOhy.psnhd,  rule=eUninstallStorage,     doc='inventory if uninstalled H2 storage unit [p.u.]')

def eInstallBoP(mDEPLOhy,p,bp):
    if bp in mDEPLOhy.gc and p == mDEPLOhy.p.first():
        return mDEPLOhy.pRatedMaxCharge[bp] * mDEPLOhy.vUnitInvestment[p,bp] == (mDEPLOhy.pBoPFactor        [bp] * pEZRatedMaxCharge * mDEPLOhy.vGlobalInvestment +
                                                                                 mDEPLOhy.pCompressionFactor[bp] * pMaxH2Inflow      * mDEPLOhy.vGlobalInvestment )
    else:
        return Constraint.Skip
mDEPLOhy.eInstallBoP            = Constraint(mDEPLOhy.pbp,    rule=eInstallBoP,           doc='BoP investment fixed to EZ investment [GW]')

GeneratingIRCTime = time.time() - StartTime
StartTime         = time.time()
print('Generating investment & retirement constraints ... ', round(GeneratingIRCTime), 's')


# Balances

def eFlowBESS(mDEPLOhy,p,sc,n,ni,nf,cc):
    if ni in s2n:
        return     sum(mDEPLOhy.vFlow[p,sc,n,ni,nf,cc] for ni,nf,cc in mDEPLOhy.la if ni in s2n) == sum( mDEPLOhy.vOutput   [p,sc,n,su] for su in mDEPLOhy.su )
    if nf in s2n:
        return     sum(mDEPLOhy.vFlow[p,sc,n,ni,nf,cc] for ni,nf,cc in mDEPLOhy.la if nf in s2n) == sum( mDEPLOhy.vESSCharge[p,sc,n,su] for su in mDEPLOhy.su )
    else:
        return Constraint.Skip
mDEPLOhy.eFlowBESS = Constraint(mDEPLOhy.psnla,                  rule=eFlowBESS,               doc='max electric flow for BESS [GW]')

def eFlowPPA(mDEPLOhy,p,sc,n,ni,nf,cc):
    if len(mDEPLOhy.ppa):
        if ni in p2n:
            return sum(mDEPLOhy.vFlow[p,sc,n,ni,nf,cc] for ni,nf,cc in mDEPLOhy.la if ni in p2n) == sum((mDEPLOhy.vOutput   [p,sc,n,ppa] - mDEPLOhy.vCurtailment[p,sc,n,ppa]) for ppa in mDEPLOhy.ppa)
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
mDEPLOhy.eFlowPPA = Constraint(mDEPLOhy.psnla,                   rule=eFlowPPA,                doc='max electric flow for PPA [GW]')

def eBalance(mDEPLOhy,p,sc,n,nd):
    if sum(1 for g in g2n[nd]) + sum(1 for nf,cc in lout[nd]) + sum(1 for ni,cc in lin[nd]) + sum(1 for ppa in p2n[nd]):
        return (sum(mDEPLOhy.vOutput     [p,sc,n,g       ] for g     in g2n [nd]) -
                sum(mDEPLOhy.vCurtailment[p,sc,n,ppa     ] for ppa   in p2n [nd]) -
                sum(mDEPLOhy.vESSCharge  [p,sc,n,es      ] for es    in e2n [nd]) +
                    mDEPLOhy.vPurchase   [p,sc,n,nd      ]                        -
                    mDEPLOhy.vSale       [p,sc,n,nd      ]                        -
                sum(mDEPLOhy.vFlow       [p,sc,n,nd,nf,cc] for nf,cc in lout[nd]) +
                sum(mDEPLOhy.vFlow       [p,sc,n,ni,nd,cc] for ni,cc in lin [nd]))   == 0.0
    else:
        return Constraint.Skip
mDEPLOhy.eBalance        = Constraint(mDEPLOhy.psnnd,            rule=eBalance,                doc='electric load balance [GW]')

def ePowerBuy(mDEPLOhy,p,sc,n,m):
    if mDEPLOhy.pMarketPeriod[p,n,m] > 0.0:
        if len(mDEPLOhy.ppa):
            return (sum(mDEPLOhy.vPurchase[p,sc,n,ndb] for ndb in mDEPLOhy.ndb) + sum(mDEPLOhy.vOutput[p,sc,n,ppa] for ppa in mDEPLOhy.ppa)) * mDEPLOhy.pMarketPeriod[p,n,m] <= mDEPLOhy.vContractedPower[p,m]
        else:
            return  sum(mDEPLOhy.vPurchase[p,sc,n,ndb] for ndb in mDEPLOhy.ndb)                                                              * mDEPLOhy.pMarketPeriod[p,n,m] <= mDEPLOhy.vContractedPower[p,m]
    else:
        return Constraint.Skip
mDEPLOhy.ePowerBuy       = Constraint(mDEPLOhy.psnm,             rule=ePowerBuy,               doc='purchase bounded by contracted power [GW]')

def ePowerSale(mDEPLOhy,p,sc,n,m):
    if mDEPLOhy.pMarketPeriod[p,n,m] > 0.0:
        return      sum(mDEPLOhy.vSale    [p,sc,n,nds] for nds in mDEPLOhy.nds)                                                              * mDEPLOhy.pMarketPeriod[p,n,m] <= mDEPLOhy.vContractedPower[p,m]
    else:
        return Constraint.Skip
mDEPLOhy.ePowerSale      = Constraint(mDEPLOhy.psnm,             rule=ePowerSale,              doc='sale bounded by contracted power [GW]')

def ePowerRule(mDEPLOhy,p,m):
    if m > mDEPLOhy.m.first():
        return mDEPLOhy.vContractedPower[p,mDEPLOhy.m.prev(m)] <= mDEPLOhy.vContractedPower[p,m]
    else:
        return Constraint.Skip
mDEPLOhy.ePowerRule      = Constraint(mDEPLOhy.pm,               rule=ePowerRule,              doc='contracted power rule [GW]')

def eMaxContractedPower(mDEPLOhy,p,m):
    if mDEPLOhy.pIndBinInvest() == 0:
        return mDEPLOhy.vContractedPower[p,m] <= (pEZRatedMaxCharge * mDEPLOhy.vGlobalInvestment) + sum(mDEPLOhy.pRatedMaxCharge[bp] * mDEPLOhy.vUnitInvestment[p,bp] for bp in mDEPLOhy.bp)
    else:
        return mDEPLOhy.vContractedPower[p,m] <=  pEZRatedMaxCharge                               + sum(mDEPLOhy.pRatedMaxCharge[bp]                                  for bp in mDEPLOhy.bp)
mDEPLOhy.eMaxContractedPower = Constraint(mDEPLOhy.pm,           rule=eMaxContractedPower,     doc='max contracted power [GW]')

def eMaxPurchaseLCF(mDEPLOhy,p,sc,n,ndb):
    if mDEPLOhy.pIndLCF() == 1 and mDEPLOhy.pIndBinInvest() != 2:
        return mDEPLOhy.vPurchase[p,sc,n,ndb] <= (mDEPLOhy.pMaxGCP[p,sc,n,ndb] * mDEPLOhy.vGlobalInvestment) + mDEPLOhy.pMaxGCPzero[p,sc,n,ndb]
    else:
        return Constraint.Skip
mDEPLOhy.eMaxPurchaseLCF     = Constraint(mDEPLOhy.psnnb,        rule=eMaxPurchaseLCF,         doc='max power purchase for LCF certification [GW]')

GeneratingEGCTime = time.time() - StartTime
StartTime         = time.time()
print('Generating electricity grid        constraints ... ', round(GeneratingEGCTime), 's')


# Storage operation constraints

def eMaxInventory(mDEPLOhy,p,sc,n,su):
    if n > mDEPLOhy.n.first():
        if su in mDEPLOhy.ec:
            return mDEPLOhy.vESSInventory[p,sc,mDEPLOhy.n.prev(n),su] <= (mDEPLOhy.pMaxStorage[p,sc,n,su] * mDEPLOhy.vUnitInvestment[p,su]) - (mDEPLOhy.pDuration[n] * (mDEPLOhy.vESSCharge2ndBlock[p,sc,n,su] + mDEPLOhy.vReserveDwC[p,sc,n,su]) * math.sqrt(mDEPLOhy.pEfficiency[su]))
        else:
            return mDEPLOhy.vESSInventory[p,sc,mDEPLOhy.n.prev(n),su] <=  mDEPLOhy.pMaxStorage[p,sc,n,su]                                   - (mDEPLOhy.pDuration[n] * (mDEPLOhy.vESSCharge2ndBlock[p,sc,n,su] + mDEPLOhy.vReserveDwC[p,sc,n,su]) * math.sqrt(mDEPLOhy.pEfficiency[su]))
    else:
        return Constraint.Skip
mDEPLOhy.eMaxInventory = Constraint(mDEPLOhy.psnsu,      rule=eMaxInventory,       doc='BESS maximum inventory [GWh]')

def eMinInventory(mDEPLOhy,p,sc,n,su):
    if n > mDEPLOhy.n.first():
        if su in mDEPLOhy.ec:
            return mDEPLOhy.vESSInventory[p,sc,mDEPLOhy.n.prev(n),su] >= (mDEPLOhy.pMinStorage[p,sc,n,su] * mDEPLOhy.vUnitInvestment[p,su]) + (mDEPLOhy.pDuration[n] * (mDEPLOhy.vOutput2ndBlock   [p,sc,n,su] + mDEPLOhy.vReserveUpD[p,sc,n,su]) / math.sqrt(mDEPLOhy.pEfficiency[su]))
        else:
            return mDEPLOhy.vESSInventory[p,sc,mDEPLOhy.n.prev(n),su] >=  mDEPLOhy.pMinStorage[p,sc,n,su]                                   + (mDEPLOhy.pDuration[n] * (mDEPLOhy.vOutput2ndBlock   [p,sc,n,su] + mDEPLOhy.vReserveUpD[p,sc,n,su]) / math.sqrt(mDEPLOhy.pEfficiency[su]))
    else:
        return Constraint.Skip
mDEPLOhy.eMinInventory = Constraint(mDEPLOhy.psnsu,      rule=eMinInventory,       doc='BESS minimum inventory [GWh]')

def eESSInventory(mDEPLOhy,p,sc,n,su):
    if   n == mDEPLOhy.n.first() and su in     mDEPLOhy.ec:
        return mDEPLOhy.vESSInventory[p,sc,n,su] + mDEPLOhy.vESSSpillage[p,sc,n,su] == mDEPLOhy.vESSIniInventory[p,sc,su]                 + (mDEPLOhy.pDuration[n] * ((math.sqrt(mDEPLOhy.pEfficiency[su]) * mDEPLOhy.vESSCharge[p,sc,n,su]) - (mDEPLOhy.vOutput[p,sc,n,su] / math.sqrt(mDEPLOhy.pEfficiency[su]))))
    elif n == mDEPLOhy.n.first() and su not in mDEPLOhy.ec:
        return mDEPLOhy.vESSInventory[p,sc,n,su] + mDEPLOhy.vESSSpillage[p,sc,n,su] == mDEPLOhy.pInitialInventory    [su]                 + (mDEPLOhy.pDuration[n] * ((math.sqrt(mDEPLOhy.pEfficiency[su]) * mDEPLOhy.vESSCharge[p,sc,n,su]) - (mDEPLOhy.vOutput[p,sc,n,su] / math.sqrt(mDEPLOhy.pEfficiency[su]))))
    elif n >  mDEPLOhy.n.first():
        return mDEPLOhy.vESSInventory[p,sc,n,su] + mDEPLOhy.vESSSpillage[p,sc,n,su] == mDEPLOhy.vESSInventory[p,sc,mDEPLOhy.n.prev(n),su] + (mDEPLOhy.pDuration[n] * ((math.sqrt(mDEPLOhy.pEfficiency[su]) * mDEPLOhy.vESSCharge[p,sc,n,su]) - (mDEPLOhy.vOutput[p,sc,n,su] / math.sqrt(mDEPLOhy.pEfficiency[su]))))
mDEPLOhy.eESSInventory = Constraint(mDEPLOhy.psnsu,      rule=eESSInventory,       doc='BESS inventory balance [GWh]')

def eInitialInventory(mDEPLOhy,p,sc,su):
    if su in mDEPLOhy.ec:
        return mDEPLOhy.vESSIniInventory[p,sc,su] / mDEPLOhy.pInitialInventory[su] == mDEPLOhy.vUnitInvestment[p,su]
    else:
        return Constraint.Skip
mDEPLOhy.eInitialInventory = Constraint(mDEPLOhy.pssu,   rule=eInitialInventory,   doc='Initial inventory for BESS candidates [p.u.]')

def eFinalInventory(mDEPLOhy,p,sc,su):
    if su in mDEPLOhy.ec:
        return mDEPLOhy.vESSIniInventory[p,sc,su] == mDEPLOhy.vESSInventory[p,sc,mDEPLOhy.n.last(),su]
    else:
        return Constraint.Skip
mDEPLOhy.eFinalInventory = Constraint(mDEPLOhy.pssu,     rule=eFinalInventory,     doc='Initial equal to final inventory for BESS candidates [GWh]')

def eChargeActivation(mDEPLOhy,p,sc,n,su):
    if mDEPLOhy.pMaxCharge[p,sc,n,su]:
        return (mDEPLOhy.vESSCharge2ndBlock[p,sc,n,su] + mDEPLOhy.vReserveDwC[p,sc,n,su]) / mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,su] <=       mDEPLOhy.vChargeInd[p,sc,n,su]
    else:
        return Constraint.Skip
mDEPLOhy.eChargeActivation    = Constraint(mDEPLOhy.psnsu, rule=eChargeActivation, doc='BESS charge activation [p.u.]')

def eTightCharge(mDEPLOhy,p,sc,n,su):
    if su in mDEPLOhy.ec:
        return mDEPLOhy.pDuration[n] * mDEPLOhy.vESSCharge[p,sc,n,su] * math.sqrt(mDEPLOhy.pEfficiency[su]) <= (mDEPLOhy.pMaxStorage[p,sc,n,su] - mDEPLOhy.pMinStorage[p,sc,n,su]) * mDEPLOhy.vUnitInvestment[p,su]
    else:
        return mDEPLOhy.pDuration[n] * mDEPLOhy.vESSCharge[p,sc,n,su] * math.sqrt(mDEPLOhy.pEfficiency[su]) <= (mDEPLOhy.pMaxStorage[p,sc,n,su] - mDEPLOhy.pMinStorage[p,sc,n,su])
mDEPLOhy.eTightCharge         = Constraint(mDEPLOhy.psnsu, rule=eTightCharge,      doc='BESS charge bound for tighter formulation [GWh]')

def eDischargeActivation(mDEPLOhy,p,sc,n,su):
    if mDEPLOhy.pMaxPower[p,sc,n,su]:
        return (mDEPLOhy.vOutput2ndBlock   [p,sc,n,su] + mDEPLOhy.vReserveUpD[p,sc,n,su]) / mDEPLOhy.pMaxPower         [p,sc,n,su] <= 1.0 - mDEPLOhy.vChargeInd[p,sc,n,su]
    else:
        return Constraint.Skip
mDEPLOhy.eDischargeActivation = Constraint(mDEPLOhy.psnsu, rule=eDischargeActivation, doc='BESS discharge activation [p.u.]')

def eTightDischarge(mDEPLOhy,p,sc,n,su):
    if su in mDEPLOhy.ec:
        return mDEPLOhy.pDuration[n] * mDEPLOhy.vOutput   [p,sc,n,su] / math.sqrt(mDEPLOhy.pEfficiency[su]) <= (mDEPLOhy.pMaxStorage[p,sc,n,su] - mDEPLOhy.pMinStorage[p,sc,n,su]) * mDEPLOhy.vUnitInvestment[p,su]
    else:
        return mDEPLOhy.pDuration[n] * mDEPLOhy.vOutput   [p,sc,n,su] / math.sqrt(mDEPLOhy.pEfficiency[su]) <= (mDEPLOhy.pMaxStorage[p,sc,n,su] - mDEPLOhy.pMinStorage[p,sc,n,su])
mDEPLOhy.eTightDischarge      = Constraint(mDEPLOhy.psnsu, rule=eTightDischarge,      doc='BESS discharge bound for tighter formulation [GWh]')

def eMaxH2Inventory(mDEPLOhy,p,sc,n,hc):
    if mDEPLOhy.pH2MaxStorage[p,sc,n,hc]:
        return mDEPLOhy.vH2Inventory[p,sc,n,hc ] / mDEPLOhy.pH2MaxStorage[p,sc,n,hc] <= mDEPLOhy.vUnitInvestment[p,hc]
    else:
        return Constraint.Skip
mDEPLOhy.eMaxH2Inventory     = Constraint(mDEPLOhy.psnhc,  rule=eMaxH2Inventory,      doc='max inventory if installed H2 storage unit [p.u.]')

def eMinH2Inventory(mDEPLOhy,p,sc,n,hc):
    if mDEPLOhy.pH2MinStorage[p,sc,n,hc]:
        return mDEPLOhy.vH2Inventory[p,sc,n,hc ] / mDEPLOhy.pH2MinStorage[p,sc,n,hc] >= mDEPLOhy.vUnitInvestment[p,hc]
    else:
        return Constraint.Skip
mDEPLOhy.eMinH2Inventory     = Constraint(mDEPLOhy.psnhc,  rule=eMinH2Inventory,      doc='min inventory if installed H2 storage unit [p.u.]')

def eH2Inventory(mDEPLOhy,p,sc,n,hs):
    if   n == mDEPLOhy.n.first() and hs in     mDEPLOhy.hc:
        return mDEPLOhy.vH2Inventory [p,sc,n,hs] + mDEPLOhy.vH2Spillage [p,sc,n,hs] == mDEPLOhy.vH2IniInventory[p,sc,hs]                 + (mDEPLOhy.pDuration[n] * ((math.sqrt(mDEPLOhy.pEfficiency[hs]) * sum(mDEPLOhy.vH2Production[p,sc,n,el] for el in mDEPLOhy.el)) - (mDEPLOhy.vH2Outflows[p,sc,n,hs] / math.sqrt(mDEPLOhy.pEfficiency[hs]))))
    elif n == mDEPLOhy.n.first() and hs not in mDEPLOhy.hc:
        return mDEPLOhy.vH2Inventory [p,sc,n,hs] + mDEPLOhy.vH2Spillage [p,sc,n,hs] == mDEPLOhy.pH2InitialInventory[hs]                  + (mDEPLOhy.pDuration[n] * ((math.sqrt(mDEPLOhy.pEfficiency[hs]) * sum(mDEPLOhy.vH2Production[p,sc,n,el] for el in mDEPLOhy.el)) - (mDEPLOhy.vH2Outflows[p,sc,n,hs] / math.sqrt(mDEPLOhy.pEfficiency[hs]))))
    elif n >  mDEPLOhy.n.first():
        return mDEPLOhy.vH2Inventory [p,sc,n,hs] + mDEPLOhy.vH2Spillage [p,sc,n,hs] == mDEPLOhy.vH2Inventory[p,sc,mDEPLOhy.n.prev(n),hs] + (mDEPLOhy.pDuration[n] * ((math.sqrt(mDEPLOhy.pEfficiency[hs]) * sum(mDEPLOhy.vH2Production[p,sc,n,el] for el in mDEPLOhy.el)) - (mDEPLOhy.vH2Outflows[p,sc,n,hs] / math.sqrt(mDEPLOhy.pEfficiency[hs]))))
mDEPLOhy.eH2Inventory        = Constraint(mDEPLOhy.psnhs,  rule=eH2Inventory,         doc='H2 storage inventory balance [tH2]')

def eInitialH2Inventory(mDEPLOhy,p,sc,hs):
    if hs in mDEPLOhy.hc:
        return mDEPLOhy.vH2IniInventory[p,sc,hs] / mDEPLOhy.pH2InitialInventory[hs] == mDEPLOhy.vUnitInvestment[p,hs]
    else:
        return Constraint.Skip
mDEPLOhy.eInitialH2Inventory = Constraint(mDEPLOhy.pshs,   rule=eInitialH2Inventory,  doc='Initial inventory for H2Storage candidates [p.u.]')

def eFinalH2Inventory(mDEPLOhy,p,sc,hs):
    if hs in mDEPLOhy.hc:
        return mDEPLOhy.vH2IniInventory[p,sc,hs] == mDEPLOhy.vH2Inventory[p,sc,mDEPLOhy.n.last(),hs]
    else:
        return Constraint.Skip
mDEPLOhy.eFinalH2Inventory = Constraint(mDEPLOhy.pshs,     rule=eFinalH2Inventory,    doc='Initial equal to final inventory for H2 storage candidates [tH2]')

GeneratingSOTime = time.time() - StartTime
StartTime        = time.time()
print('Generating storage operation       constraints ... ', round(GeneratingSOTime), 's')


# Balancing capacity and energy constraints

def eMaxOutput2ndBlock(mDEPLOhy,p,sc,n,su):
    if su in mDEPLOhy.ec:
        return (mDEPLOhy.vOutput2ndBlock[p,sc,n,su] + mDEPLOhy.vReserveUpD[p,sc,n,su]) / mDEPLOhy.pMaxPower[p,sc,n,su] <= mDEPLOhy.vUnitInvestment[p,su]
    else:
        return (mDEPLOhy.vOutput2ndBlock[p,sc,n,su] + mDEPLOhy.vReserveUpD[p,sc,n,su]) / mDEPLOhy.pMaxPower[p,sc,n,su] <= 1.0
mDEPLOhy.eMaxOutput2ndBlock       = Constraint(mDEPLOhy.psnsu, rule=eMaxOutput2ndBlock,    doc='max output second block of a BESS unit [p.u.]')

def eMinOutput2ndBlock(mDEPLOhy,p,sc,n,su):
    if mDEPLOhy.pIndReserves() == 1:
        return (mDEPLOhy.vOutput2ndBlock[p,sc,n,su] - mDEPLOhy.vReserveDwD[p,sc,n,su]) / mDEPLOhy.pMaxPower[p,sc,n,su] >= 0.0
    else:
        return Constraint.Skip
mDEPLOhy.eMinOutput2ndBlock       = Constraint(mDEPLOhy.psnsu, rule=eMinOutput2ndBlock,    doc='min output second block of a BESS unit [p.u.]')

def eBESSTotalOutput(mDEPLOhy,p,sc,n,su):
    return mDEPLOhy.vOutput[p,sc,n,su] == mDEPLOhy.vOutput2ndBlock[p,sc,n,su] + (mDEPLOhy.vReserveUpD[p,sc,n,su] * mDEPLOhy.pActivationUp[p,sc,n]) - (mDEPLOhy.vReserveDwD[p,sc,n,su] * mDEPLOhy.pActivationDw[p,sc,n])
mDEPLOhy.eBESSTotalOutput         = Constraint(mDEPLOhy.psnsu, rule=eBESSTotalOutput,      doc='total output of a BESS unit [GW]')

def eReserveCMaxRatioDwUp(mDEPLOhy,p,sc,n,nr):
    if mDEPLOhy.pIndReserves() == 1:
        return  mDEPLOhy.vReserveDwC[p,sc,n,nr] <= mDEPLOhy.pMaxRatioDwUp[p,sc,n] * mDEPLOhy.vReserveUpC[p,sc,n,nr]
    else:
        return Constraint.Skip
mDEPLOhy.eReserveCMaxRatioDwUp    = Constraint(mDEPLOhy.psnnr, rule=eReserveCMaxRatioDwUp, doc='max reserve dw when    charging [GW]')

def eReserveDMaxRatioDwUp(mDEPLOhy,p,sc,n,su):
    if mDEPLOhy.pIndReserves() == 1:
        return  mDEPLOhy.vReserveDwD[p,sc,n,su] <= mDEPLOhy.pMaxRatioDwUp[p,sc,n] * mDEPLOhy.vReserveUpD[p,sc,n,su]
    else:
        return Constraint.Skip
mDEPLOhy.eReserveDMaxRatioDwUp    = Constraint(mDEPLOhy.psnsu, rule=eReserveDMaxRatioDwUp, doc='max reserve dw when discharging [GW]')

def eReserveUpIfEnergy(mDEPLOhy,p,sc,n,su):
    if mDEPLOhy.pIndReserves() == 1:
        if su in mDEPLOhy.ec:
            return (mDEPLOhy.vOutput2ndBlock[p,sc,n,su] + mDEPLOhy.vReserveUpD[p,sc,n,su]) / math.sqrt(mDEPLOhy.pEfficiency[su]) <= (mDEPLOhy.vESSInventory[p,sc,n,su] - (mDEPLOhy.pMinStorage[p,sc,n,su] * mDEPLOhy.vUnitInvestment[p,su])) / mDEPLOhy.pDuration[n]
        else:
            return (mDEPLOhy.vOutput2ndBlock[p,sc,n,su] + mDEPLOhy.vReserveUpD[p,sc,n,su]) / math.sqrt(mDEPLOhy.pEfficiency[su]) <= (mDEPLOhy.vESSInventory[p,sc,n,su] -  mDEPLOhy.pMinStorage[p,sc,n,su]                                  ) / mDEPLOhy.pDuration[n]
    else:
        return Constraint.Skip
mDEPLOhy.eReserveUpIfEnergy       = Constraint(mDEPLOhy.psnsu, rule=eReserveUpIfEnergy,    doc='up reserve if energy available [GW]')

def eReserveDwIfEnergy(mDEPLOhy,p,sc,n,su):
    if mDEPLOhy.pIndReserves() == 1:
        if su in mDEPLOhy.ec:
            return (mDEPLOhy.vESSCharge2ndBlock[p,sc,n,su] + mDEPLOhy.vReserveDwC[p,sc,n,su]) * math.sqrt(mDEPLOhy.pEfficiency[su]) <= ((mDEPLOhy.pMaxStorage[p,sc,n,su] * mDEPLOhy.vUnitInvestment[p,su]) - mDEPLOhy.vESSInventory[p,sc,n,su]) / mDEPLOhy.pDuration[n]
        else:
            return (mDEPLOhy.vESSCharge2ndBlock[p,sc,n,su] + mDEPLOhy.vReserveDwC[p,sc,n,su]) * math.sqrt(mDEPLOhy.pEfficiency[su]) <= ( mDEPLOhy.pMaxStorage[p,sc,n,su]                                   - mDEPLOhy.vESSInventory[p,sc,n,su]) / mDEPLOhy.pDuration[n]
    else:
        return Constraint.Skip
mDEPLOhy.eReserveDwIfEnergy       = Constraint(mDEPLOhy.psnsu, rule=eReserveDwIfEnergy,    doc='dw reserve if energy available [GW]')

def eMaxEnergyUpD(mDEPLOhy,p,sc,n,su):
    if mDEPLOhy.pIndReserves() == 1:
        return (mDEPLOhy.vReserveUpD[p,sc,n,su] * mDEPLOhy.pActivationUp[p,sc,n]) / mDEPLOhy.pMaxPower         [p,sc,n,su] <=        mDEPLOhy.vActivationD[p,sc,n,su]
    else:
        return Constraint.Skip
mDEPLOhy.eMaxEnergyUpD            = Constraint(mDEPLOhy.psnsu, rule=eMaxEnergyUpD,         doc='max energy up when discharging [p.u.]')

def eMaxEnergyUpC(mDEPLOhy,p,sc,n,nr):
    if mDEPLOhy.pIndReserves() == 1:
        return (mDEPLOhy.vReserveUpC[p,sc,n,nr] * mDEPLOhy.pActivationUp[p,sc,n]) / mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,nr] <=        mDEPLOhy.vActivationC[p,sc,n,nr]
    else:
        return Constraint.Skip
mDEPLOhy.eMaxEnergyUpC            = Constraint(mDEPLOhy.psnnr, rule=eMaxEnergyUpC,         doc='max energy up when charging [p.u.]')

def eMaxEnergyDwD(mDEPLOhy,p,sc,n,su):
    if mDEPLOhy.pIndReserves() == 1:
        return (mDEPLOhy.vReserveDwD[p,sc,n,su] * mDEPLOhy.pActivationDw[p,sc,n]) / mDEPLOhy.pMaxPower         [p,sc,n,su] <= (1.0 - mDEPLOhy.vActivationD[p,sc,n,su])
    else:
        return Constraint.Skip
mDEPLOhy.eMaxEnergyDwD            = Constraint(mDEPLOhy.psnsu, rule=eMaxEnergyDwD,         doc='max energy dw when discharging [p.u.]')

def eMaxEnergyDwC(mDEPLOhy,p,sc,n,nr):
    if mDEPLOhy.pIndReserves() == 1:
        return (mDEPLOhy.vReserveDwC[p,sc,n,nr] * mDEPLOhy.pActivationDw[p,sc,n]) / mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,nr] <= (1.0 - mDEPLOhy.vActivationC[p,sc,n,nr])
    else:
        return Constraint.Skip
mDEPLOhy.eMaxEnergyDwC            = Constraint(mDEPLOhy.psnnr, rule=eMaxEnergyDwC,         doc='max energy dw when charging [p.u.]')

GeneratingORTime = time.time() - StartTime
StartTime        = time.time()
print('Generating balancing reserves      constraints ... ', round(GeneratingORTime), 's')


# Consumption constraints

def eCommMaxCharge2ndBlock(mDEPLOhy,p,sc,n,el):
    return     (mDEPLOhy.vESSCharge2ndBlock[p,sc,n,el] + mDEPLOhy.vReserveDwC[p,sc,n,el]) / mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,el] <= mDEPLOhy.vCommitment[p,sc,n,el]
mDEPLOhy.eCommMaxCharge2ndBlock    = Constraint(mDEPLOhy.psnel, rule=eCommMaxCharge2ndBlock,    doc='max charge second block of a commited EZ unit [p.u.]')

def eMaxCharge2ndBlock(mDEPLOhy,p,sc,n,nr):
    if nr in mDEPLOhy.ec:
        return (mDEPLOhy.vESSCharge2ndBlock[p,sc,n,nr] + mDEPLOhy.vReserveDwC[p,sc,n,nr]) / mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,nr] <= mDEPLOhy.vUnitInvestment[p,nr]
    else:
        return (mDEPLOhy.vESSCharge2ndBlock[p,sc,n,nr] + mDEPLOhy.vReserveDwC[p,sc,n,nr]) / mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,nr] <= 1.0
mDEPLOhy.eMaxCharge2ndBlock        = Constraint(mDEPLOhy.psnnr, rule=eMaxCharge2ndBlock,        doc='max charge second block of an ESS unit [p.u.]')

def eMinCharge2ndBlock(mDEPLOhy,p,sc,n,nr):
    if mDEPLOhy.pIndReserves() == 1:
        return (mDEPLOhy.vESSCharge2ndBlock[p,sc,n,nr] - mDEPLOhy.vReserveUpC[p,sc,n,nr]) / mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,nr] >= 0.0
    else:
        return Constraint.Skip
mDEPLOhy.eMinCharge2ndBlock        = Constraint(mDEPLOhy.psnnr, rule=eMinCharge2ndBlock,        doc='min charge second block of an ESS unit [p.u.]')

def eESSTotalCharge(mDEPLOhy,p,sc,n,nr):
    if nr in mDEPLOhy.el:
        return mDEPLOhy.vESSCharge[p,sc,n,nr] == (mDEPLOhy.pMinCharge[p,sc,n,nr] * mDEPLOhy.vCommInvestment[p,sc,n,nr]) + mDEPLOhy.vESSCharge2ndBlock[p,sc,n,nr] + (mDEPLOhy.vReserveDwC[p,sc,n,nr] * mDEPLOhy.pActivationDw[p,sc,n]) - (mDEPLOhy.vReserveUpC[p,sc,n,nr] * mDEPLOhy.pActivationUp[p,sc,n])
    else:
        return mDEPLOhy.vESSCharge[p,sc,n,nr] ==                                                                          mDEPLOhy.vESSCharge2ndBlock[p,sc,n,nr] + (mDEPLOhy.vReserveDwC[p,sc,n,nr] * mDEPLOhy.pActivationDw[p,sc,n]) - (mDEPLOhy.vReserveUpC[p,sc,n,nr] * mDEPLOhy.pActivationUp[p,sc,n])
mDEPLOhy.eESSTotalCharge = Constraint(mDEPLOhy.psnnr,          rule=eESSTotalCharge,             doc='total charge of an ESS unit [GW]')

def eBoPConsumption(mDEPLOhy,p,sc,n,bp):
    return mDEPLOhy.vESSCharge[p,sc,n,bp] == (mDEPLOhy.pCompressionFactor[bp] * sum(mDEPLOhy.vH2Production[p,sc,n,el] for el in mDEPLOhy.el) +
                                              mDEPLOhy.pBoPFactor        [bp] * sum(mDEPLOhy.vESSCharge   [p,sc,n,el] for el in mDEPLOhy.el) +
                                              mDEPLOhy.pSBConsumption    [bp] * sum(mDEPLOhy.vSBInvestment[p,sc,n,el] for el in mDEPLOhy.el) )
mDEPLOhy.eBoPConsumption = Constraint(mDEPLOhy.psnbp,         rule=eBoPConsumption,              doc='BoP consumption [GW]')

GeneratingCCTime = time.time() - StartTime
StartTime        = time.time()
print('Generating consumption   operation constraints ... ', round(GeneratingCCTime), 's')


# Linearization constraints

def eCommInvestment1(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vCommInvestment[p,sc,n,el] <= mDEPLOhy.vCommitment[p,sc,n,el]
    else:
        return mDEPLOhy.vCommInvestment[p,sc,n,el] == mDEPLOhy.vCommitment[p,sc,n,el]
mDEPLOhy.eCommInvestment1 = Constraint(mDEPLOhy.psnel, rule=eCommInvestment1, doc='Linearization of EZ investment and commitment 1 [p.u.]')

def eCommInvestment2(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vCommInvestment[p,sc,n,el] - mDEPLOhy.vUnitInvestment[p,el] <= 0.0
    else:
        return Constraint.Skip
mDEPLOhy.eCommInvestment2 = Constraint(mDEPLOhy.psnel, rule=eCommInvestment2, doc='Linearization of EZ investment and commitment 2 [p.u.]')

def eCommInvestment3(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vUnitInvestment[p,el] - mDEPLOhy.vCommInvestment[p,sc,n,el] + mDEPLOhy.vCommitment[p,sc,n,el] <= 1.0
    else:
        return Constraint.Skip
mDEPLOhy.eCommInvestment3 = Constraint(mDEPLOhy.psnel, rule=eCommInvestment3, doc='Linearization of EZ investment and commitment 3 [p.u.]')

def eSUInvestment1(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vSUInvestment[p,sc,n,el] <= mDEPLOhy.vStartUp[p,sc,n,el]
    else:
        return mDEPLOhy.vSUInvestment[p,sc,n,el] == mDEPLOhy.vStartUp[p,sc,n,el]
mDEPLOhy.eSUInvestment1   = Constraint(mDEPLOhy.psnel, rule=eSUInvestment1,   doc='Linearization of EZ investment and SU 1 [p.u.]')

def eSUInvestment2(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vSUInvestment[p,sc,n,el] - mDEPLOhy.vUnitInvestment[p,el] <= 0.0
    else:
        return Constraint.Skip
mDEPLOhy.eSUInvestment2   = Constraint(mDEPLOhy.psnel, rule=eSUInvestment2,   doc='Linearization of EZ investment and SU 2 [p.u.]')

def eSUInvestment3(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vUnitInvestment[p,el] - mDEPLOhy.vSUInvestment[p,sc,n,el] + mDEPLOhy.vStartUp[p,sc,n,el] <= 1.0
    else:
        return Constraint.Skip
mDEPLOhy.eSUInvestment3 =   Constraint(mDEPLOhy.psnel, rule=eSUInvestment3,   doc='Linearization of EZ investment and SU 3 [p.u.]')

def eSDInvestment1(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vSDInvestment[p,sc,n,el] <= mDEPLOhy.vShutDown[p,sc,n,el]
    else:
        return mDEPLOhy.vSDInvestment[p,sc,n,el] == mDEPLOhy.vShutDown[p,sc,n,el]
mDEPLOhy.eSDInvestment1   = Constraint(mDEPLOhy.psnel, rule=eSDInvestment1,   doc='Linearization of EZ investment and SD 1 [p.u.]')

def eSDInvestment2(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vSDInvestment[p,sc,n,el] - mDEPLOhy.vUnitInvestment[p,el] <= 0.0
    else:
        return Constraint.Skip
mDEPLOhy.eSDInvestment2   = Constraint(mDEPLOhy.psnel, rule=eSDInvestment2,   doc='Linearization of EZ investment and SD 2 [p.u.]')

def eSDInvestment3(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vUnitInvestment[p,el] - mDEPLOhy.vSDInvestment[p,sc,n,el] + mDEPLOhy.vShutDown[p,sc,n,el] <= 1.0
    else:
        return Constraint.Skip
mDEPLOhy.eSDInvestment3 =   Constraint(mDEPLOhy.psnel, rule=eSDInvestment3,   doc='Linearization of EZ investment and SD 3 [p.u.]')

def eSBInvestment1(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vSBInvestment[p,sc,n,el] <= mDEPLOhy.vStandBy[p,sc,n,el]
    else:
        return mDEPLOhy.vSBInvestment[p,sc,n,el] == mDEPLOhy.vStandBy[p,sc,n,el]
mDEPLOhy.eSBInvestment1   = Constraint(mDEPLOhy.psnel, rule=eSBInvestment1,   doc='Linearization of EZ investment and SB 1 [p.u.]')

def eSBInvestment2(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vSBInvestment[p,sc,n,el] - mDEPLOhy.vUnitInvestment[p,el] <= 0.0
    else:
        return Constraint.Skip
mDEPLOhy.eSBInvestment2   = Constraint(mDEPLOhy.psnel, rule=eSBInvestment2,   doc='Linearization of EZ investment and SB 2 [p.u.]')

def eSBInvestment3(mDEPLOhy,p,sc,n,el):
    if (mDEPLOhy.pIndBinInvest() == 0 or mDEPLOhy.pIndBinUnitInvest[el] == 0) and el in mDEPLOhy.ec:
        return mDEPLOhy.vUnitInvestment[p,el] - mDEPLOhy.vSBInvestment[p,sc,n,el] + mDEPLOhy.vStandBy[p,sc,n,el] <= 1.0
    else:
        return Constraint.Skip
mDEPLOhy.eSBInvestment3 =   Constraint(mDEPLOhy.psnel, rule=eSBInvestment3,   doc='Linearization of EZ investment and SB 3 [p.u.]')

GeneratingLZTime = time.time() - StartTime
StartTime        = time.time()
print('Generating linearization           constraints ... ', round(GeneratingLZTime), 's')


# Electrolyzer constraints

def eMaxCommRFNBO(mDEPLOhy,p):
    if mDEPLOhy.pIndRFNBO() == 1 and mDEPLOhy.pAPRE[p] > 0.9:
        return sum(mDEPLOhy.pScenProb[p,sc] * mDEPLOhy.vCommitment[p,sc,n,el] * mDEPLOhy.pDuration[n] for sc in PeriodScen[p] for n in mDEPLOhy.n for el in mDEPLOhy.el) <= mDEPLOhy.pNRH[p]
    else:
        return Constraint.Skip
mDEPLOhy.eMaxCommRFNBO       = Constraint(mDEPLOhy.p    ,    rule=eMaxCommRFNBO,          doc='max operation hours for RFNBO certification [h]')

def eHydrogenProduction(mDEPLOhy,p,sc,n,el):
    if mDEPLOhy.pIndPiecewise() == 1:
        return mDEPLOhy.vH2Production[p,sc,n,el] == (mDEPLOhy.pMinH2Prod[p] * mDEPLOhy.vCommInvestment[p,sc,n,el]) +   mDEPLOhy.vH2Prod2ndBlock   [p,sc,n,el]
    else:
        return mDEPLOhy.vH2Production[p,sc,n,el] == (mDEPLOhy.pMinH2Prod[p] * mDEPLOhy.vCommInvestment[p,sc,n,el]) + ((mDEPLOhy.vESSCharge2ndBlock[p,sc,n,el] + (mDEPLOhy.vReserveDwC[p,sc,n,el] * mDEPLOhy.pActivationDw[p,sc,n]) - (mDEPLOhy.vReserveUpC[p,sc,n,el] * mDEPLOhy.pActivationUp[p,sc,n])) * mDEPLOhy.pH2ProdRate[p])
mDEPLOhy.eHydrogenProduction = Constraint(mDEPLOhy.psnel,    rule=eHydrogenProduction,    doc='hydrogen production [tonH2/h]')

def ePiecewiseProd(mDEPLOhy,p,sc,n,el,cl):
    if mDEPLOhy.pIndPiecewise() == 1:
        if el in mDEPLOhy.ec:
            return mDEPLOhy.vH2Prod2ndBlock[p,sc,n,el] <= ((mDEPLOhy.vESSCharge2ndBlock[p,sc,n,el] + (mDEPLOhy.vReserveDwC[p,sc,n,el] * mDEPLOhy.pActivationDw[p,sc,n]) - (mDEPLOhy.vReserveUpC[p,sc,n,el] * mDEPLOhy.pActivationUp[p,sc,n])) * mDEPLOhy.pSegH2ProdRate[p,cl]) + (mDEPLOhy.pSegIntercept[p,cl] * mDEPLOhy.vCommInvestment[p,sc,n,el])
        else:
            return mDEPLOhy.vH2Prod2ndBlock[p,sc,n,el] <= ((mDEPLOhy.vESSCharge2ndBlock[p,sc,n,el] + (mDEPLOhy.vReserveDwC[p,sc,n,el] * mDEPLOhy.pActivationDw[p,sc,n]) - (mDEPLOhy.vReserveUpC[p,sc,n,el] * mDEPLOhy.pActivationUp[p,sc,n])) * mDEPLOhy.pSegH2ProdRate[p,cl]) + (mDEPLOhy.pSegIntercept[p,cl] * mDEPLOhy.vCommitment    [p,sc,n,el])
    else:
        return Constraint.Skip
mDEPLOhy.ePiecewiseProd      = Constraint(mDEPLOhy.psnel,mDEPLOhy.cl, rule=ePiecewiseProd, doc='piecewise prod 2nd block [tonH2/h]')

def eProdOutflowsRampUp(mDEPLOhy,p,sc,n,hs):
    if n > mDEPLOhy.n.first() and mDEPLOhy.pOutflowsRampUp[hs] > 0.0:
        return (mDEPLOhy.vH2Outflows[p,sc,n,hs] - mDEPLOhy.vH2Outflows[p,sc,mDEPLOhy.n.prev(n),hs]) <= mDEPLOhy.pOutflowsRampUp[hs]
    else:
        return Constraint.Skip
mDEPLOhy.eProdOutflowsRampUp = Constraint(mDEPLOhy.psnhs,    rule=eProdOutflowsRampUp,    doc='Product outflows ramp up [tonH2/h]')

def eProdOutflowsRampDw(mDEPLOhy,p,sc,n,hs):
    if n > mDEPLOhy.n.first() and mDEPLOhy.pOutflowsRampDw[hs] < 0.0:
        return (mDEPLOhy.vH2Outflows[p,sc,n,hs] - mDEPLOhy.vH2Outflows[p,sc,mDEPLOhy.n.prev(n),hs]) >= mDEPLOhy.pOutflowsRampDw[hs]
    else:
        return Constraint.Skip
mDEPLOhy.eProdOutflowsRampDw = Constraint(mDEPLOhy.psnhs,    rule=eProdOutflowsRampDw,    doc='Product outflows ramp down [tonH2/h]')

def eOperationalStates(mDEPLOhy,p,sc,n,el):
    if mDEPLOhy.pInitialPeriod[el] <= p < mDEPLOhy.pFinalPeriod[el]:
        return mDEPLOhy.vCommitment[p,sc,n,el] + mDEPLOhy.vStandBy[p,sc,n,el] + mDEPLOhy.vOff[p,sc,n,el] == 1.0
    else:
        return Constraint.Skip
mDEPLOhy.eOperationalStates =Constraint(mDEPLOhy.psnel,      rule=eOperationalStates,     doc='EZ operational states balance [p.u.]')

def eStartUp(mDEPLOhy,p,sc,n,el):
    if n > mDEPLOhy.n.first() and mDEPLOhy.pInitialPeriod[el] <= p < mDEPLOhy.pFinalPeriod[el] and mDEPLOhy.pIndTwoStates() == 0:
        return mDEPLOhy.vStartUp[p,sc,n,el] >= mDEPLOhy.vCommitment[p,sc,n,el] - mDEPLOhy.vCommitment[p,sc,mDEPLOhy.n.prev(n),el] - mDEPLOhy.vStandBy[p,sc,mDEPLOhy.n.prev(n),el]
    else:
        return Constraint.Skip
mDEPLOhy.eStartUp           =Constraint(mDEPLOhy.psnel,      rule=eStartUp,               doc='EZ start-up [p.u.]')

def eShutDown(mDEPLOhy,p,sc,n,el):
    if n > mDEPLOhy.n.first() and mDEPLOhy.pInitialPeriod[el] <= p < mDEPLOhy.pFinalPeriod[el] and mDEPLOhy.pIndTwoStates() == 0:
        return mDEPLOhy.vShutDown[p,sc,n,el] >= mDEPLOhy.vOff[p,sc,n,el] - mDEPLOhy.vOff[p,sc,mDEPLOhy.n.prev(n),el] - mDEPLOhy.vStandBy[p,sc,mDEPLOhy.n.prev(n),el]
    else:
        return Constraint.Skip
mDEPLOhy.eShutDown          =Constraint(mDEPLOhy.psnel,      rule=eShutDown,              doc='EZ shutdown [p.u.]')

def eSBtoOFF(mDEPLOhy,p,sc,n,el):
    if n > mDEPLOhy.n.first() and mDEPLOhy.pInitialPeriod[el] <= p < mDEPLOhy.pFinalPeriod[el] and mDEPLOhy.pIndTwoStates() == 0:
        return mDEPLOhy.vOff[p,sc,n,el] + mDEPLOhy.vStandBy[p,sc,mDEPLOhy.n.prev(n),el] <= 1.0
    else:
        return Constraint.Skip
mDEPLOhy.eSBtoOFF           =Constraint(mDEPLOhy.psnel,      rule=eSBtoOFF,               doc='No transition allowed from SB to OFF [p.u.]')

def eOFFtoSB(mDEPLOhy,p,sc,n,el):
    if n > mDEPLOhy.n.first() and mDEPLOhy.pInitialPeriod[el] <= p < mDEPLOhy.pFinalPeriod[el] and mDEPLOhy.pIndTwoStates() == 0:
        return mDEPLOhy.vOff[p,sc,mDEPLOhy.n.prev(n),el] + mDEPLOhy.vStandBy[p,sc,n,el] <= 1.0
    else:
        return Constraint.Skip
mDEPLOhy.eOFFtoSB           =Constraint(mDEPLOhy.psnel,      rule=eOFFtoSB,               doc='No transition allowed from OFF to SB [p.u.]')

def ePeriodOutflows(mDEPLOhy,p,sc):
    return mDEPLOhy.vPeriodOutflows[p,sc]   == sum(mDEPLOhy.pPeriodWeight[p] * mDEPLOhy.pDuration[n] * mDEPLOhy.vH2Outflows[p,sc,n,hs] for n in mDEPLOhy.n for hs in mDEPLOhy.hs)
mDEPLOhy.ePeriodOutflows    =  Constraint(mDEPLOhy.ps,       rule=ePeriodOutflows,        doc='periodic hydrogen outflows [tonH2]')

def eTotalOutflows(mDEPLOhy):
    return mDEPLOhy.vTotalOutflows         == sum(                                                          mDEPLOhy.vPeriodOutflows[p,sc] for p in mDEPLOhy.p for sc in PeriodScen[p]) / (len(mDEPLOhy.sc)/len(mDEPLOhy.p))
mDEPLOhy.eTotalOutflows     =  Constraint(                   rule=eTotalOutflows,         doc='total system outflows [tonH2]')

def eDiscTotalOutflows(mDEPLOhy):
    return mDEPLOhy.vDiscTotalOutflows     == sum(mDEPLOhy.pDiscountFactor[p] / mDEPLOhy.pPeriodWeight[p] * mDEPLOhy.vPeriodOutflows[p,sc] for p in mDEPLOhy.p for sc in PeriodScen[p]) / (len(mDEPLOhy.sc)/len(mDEPLOhy.p))
mDEPLOhy.eDiscTotalOutflows = Constraint(                    rule=eDiscTotalOutflows,     doc='discounted total system outflows [tonH2]')

def eMinCapFactor(mDEPLOhy,p,sc):
    if mDEPLOhy.pIndCapFactor() == 1:
        if el in mDEPLOhy.ec and sc in PeriodScen[p]:
            return sum(sum((mDEPLOhy.pDuration[n] * mDEPLOhy.vESSCharge[p,sc,n,el]) for n in mDEPLOhy.n) for el in mDEPLOhy.el) >= mDEPLOhy.pMinCapFactor * pEZRatedMaxCharge * mDEPLOhy.vGlobalInvestment * len(mDEPLOhy.nn)
        else:
            return sum(sum((mDEPLOhy.pDuration[n] * mDEPLOhy.vESSCharge[p,sc,n,el]) for n in mDEPLOhy.n) for el in mDEPLOhy.el) >= mDEPLOhy.pMinCapFactor * pEZRatedMaxCharge *                              len(mDEPLOhy.nn)
    else:
        return Constraint.Skip
mDEPLOhy.eMinCapFactor    =  Constraint(mDEPLOhy.ps,         rule=eMinCapFactor,          doc='min capacity factor [GWh]')

GeneratingECTime = time.time() - StartTime
StartTime        = time.time()
print('Generating electrolyzer  operation constraints ... ', round(GeneratingECTime), 's')


# =========================
# Solving the problem
# =========================

Solver = SolverFactory(SolverName)

if SolverName == 'gurobi':
    # Define log file
    gurobi_log_path = _path + '/DEPLOhy_log_' + CaseName + '.txt'

    # Initialise containers
    perf_summary     = {}
    convergence_data = []

    # Create lp-format file
    mDEPLOhy.write(_path + '/DEPLOhy_' + CaseName + '.lp',io_options={'symbolic_solver_labels': True})

    # Solver options
    Solver.options['OutputFlag'      ] = 1
    Solver.options['DisplayInterval' ] = 100
    Solver.options['LPWarmStart'     ] = 2
    Solver.options['Method'          ] = 2
    Solver.options['Crossover'       ] = -1
    Solver.options['MIPGap'          ] = 0.01
    Solver.options['Threads'         ] = int((psutil.cpu_count(logical=True) + psutil.cpu_count(logical=False))/2)
    Solver.options['TimeLimit'       ] = 43200
    Solver.options['IterationLimit'  ] = 36000000

    # Solve + timing
    start_time                       = time.time()
    SolverResults                    = Solver.solve(mDEPLOhy, tee=True, logfile=gurobi_log_path) # tee=True displays the output of the solver
    end_time                         = time.time()
    perf_summary['Solver time [s]']  = round(end_time - start_time, 2)
    SolverResults.write()                                                                        # summary of the solver results


    # Extract additional metrics from Gurobi log
    with open(gurobi_log_path, 'r') as f:
        for line in f:
            if 'Best objective' in line:
                try:
                    obj = regex.search(r'Best objective\s+([-\d.eE+]+)', line)
                    bnd = regex.search(r'best bound\s+([-\d.eE+]+)'    , line)
                    gap = regex.search(r'gap\s+([-\d.eE+]+)%'          , line)
                    if obj:
                        perf_summary['ObjValue'] = float(obj.group(1))
                    if bnd:
                        perf_summary['ObjBound'] = float(bnd.group(1))
                    if gap:
                        perf_summary['MIPGap'  ] = float(gap.group(1))
                except:
                    pass
            elif 'Explored' in line and 'nodes' in line:
                try:
                    perf_summary['NodeCount'] = int(line.split()[1])
                except:
                    pass
            elif 'Optimal solution found' in line:
                perf_summary['Status'] = 'Optimal'
            elif 'Time limit reached' in line:
                perf_summary['Status'] = 'TimeLimit'

    # Model metrics
    n_bin  = 0
    n_int  = 0
    n_cont = 0
    for v in mDEPLOhy.component_data_objects(pyo.Var, active=True):
        if v.is_binary():
            n_bin  += 1
        elif v.is_integer():
            n_int  += 1
        else:
            n_cont += 1
    perf_summary.update({
        'nPeriods'       : len(mDEPLOhy.p),
        'nTimeSteps'     : len(mDEPLOhy.n),
        'nScenarios'     : len(mDEPLOhy.sc),
        'nVariables'     : mDEPLOhy.nvariables(),
        'nFixedVar'      : mDEPLOhy.nFixedVariables(),
        'nBinaryVar'     : n_bin,
        'nIntegerVar'    : n_int,
        'nContinuousVar' : n_cont,
        'nConstraints'   : mDEPLOhy.nconstraints()})

    # System info
    system_info = {}
    system_info['Machine'] = socket.gethostname()
    system_info['OS'     ] = f"{platform.system()} {platform.release()}"
    system_info['CPU'    ] = platform.processor()
    system_info['Cores'  ] = psutil.cpu_count(logical=False)
    freq = psutil.cpu_freq()
    if freq:
        system_info['CPU_freq_GHz'] = round(freq.max/1000, 2)
    system_info['RAM_GB'] = round(psutil.virtual_memory().total/1e9, 1)
    system_summary = (
        f"{system_info['Machine']} "
        f"[{system_info['OS'    ]}; "
        f"{system_info['Cores'  ]} cores; "
        f"{system_info['CPU'    ]} at {system_info.get('CPU_freq_GHz', 'N/A')} GHz; "
        f"{system_info['RAM_GB' ]} GB]")


    # Convergence
    t0 = None
    with open(gurobi_log_path, 'r') as f:
        for line in f:
            if '%' not in line:
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                is_heuristic = parts[0] == 'H'
                is_star      = parts[0] == '*'
                if is_heuristic or is_star:
                    parts = parts[1:]
                time_str = parts[-1]
                if not time_str.endswith('s'):
                    continue
                time_val = float(time_str.replace('s',''))
                if t0 is None:
                    t0 = time_val
                cpu_time  = time_val - t0
                gap_str   = next(p for p in parts if '%' in p)
                gap       = float(gap_str.replace('%', ''))
                gap_index = parts.index(gap_str)
                incumbent = float(parts[gap_index - 2])
                bound     = float(parts[gap_index - 1])
                try:
                    node = int(parts[0])
                except:
                    node = None
                convergence_data.append((cpu_time, node, incumbent, bound, gap))
            except:
                continue
    convergence_df         = pd.DataFrame(convergence_data, columns=['CPU time [s]', 'Node', 'Incumbent', 'BestBound', 'Gap%'])
    convergence_df['Node'] = convergence_df['Node'].ffill()
    print(f'  convergence points extracted: {len(convergence_df)}')


SolvingTime = time.time() - StartTime
StartTime   = time.time()

print('******** Solution for scenario ' + str(sc) + ' ********')
print('  Problem size                  ... ', mDEPLOhy.model().nconstraints(), 'constraints,', mDEPLOhy.model().nvariables() - mDEPLOhy.nFixedVariables + 1, 'variables')
print('  Solution time                 ... ', round(SolvingTime/60, ndigits=2), 'min')
print('  Total H2 production       [tonH2] ', round( mDEPLOhy.vTotalOutflows()                                          , ndigits=1))
print('  LCOH                   [EUR/kgH2] ', round((mDEPLOhy.vTotalSCost()/mDEPLOhy.vDiscTotalOutflows())*1e3          , ndigits=2))
print('  Total system          cost [MEUR] ', round( mDEPLOhy.vTotalSCost()                                             , ndigits=3))
print('  Total fixed capital   cost [MEUR] ', round( mDEPLOhy.vTotalFCost()                                             , ndigits=3))
print('  Total operational     cost [MEUR] ', round( mDEPLOhy.vTotalOCost()                                             , ndigits=3))
print('  Total power market    cost [MEUR] ', round( mDEPLOhy.vTotalMCost()                                             , ndigits=3))
print('  Total power market revenue [MEUR] ', round( mDEPLOhy.vTotalMRevenue()                                          , ndigits=3))
print('********************************************')


# =========================
# Output results
# =========================

# Performance data

if SolverName == 'gurobi':
    # System info
    pd.DataFrame([system_info]).to_csv(_path + '/DEPLOhy_Solution_01_SystemInfo_' + CaseName + '.csv', index=False)

    # Performance
    pd.DataFrame([perf_summary]).to_csv(_path + '/DEPLOhy_Solution_02_Performance_' + CaseName + '.csv',index=False)

    # Convergence
    if len(convergence_df) > 0:
        convergence_df.to_csv(_path + '/DEPLOhy_Solution_03_Convergence_' + CaseName + '.csv', index=False)
        plt.figure()
        plt.plot(convergence_df['CPU time [s]'], convergence_df['Incumbent'], label='Incumbent')
        plt.plot(convergence_df['CPU time [s]'], convergence_df['BestBound'], label='Best bound')
        plt.xlabel('CPU time [s]')
        plt.ylabel('Objective value')
        plt.title('Convergence evolution')
        plt.legend()
        plt.grid()
        plt.savefig(_path + '/DEPLOhy_Solution_04_ConvergencePlot_' + CaseName + '.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("[!] No convergence data found in log")


# Global LCOH
CAPEX       = pd.Series(data=[    mDEPLOhy.vTotalFCost()                                                                  /mDEPLOhy.vDiscTotalOutflows()* 1e3])
OPEX_NonEn  = pd.Series(data=[sum(mDEPLOhy.vUnitOpCost[p,g]() for p,g in mDEPLOhy.pg if g not in mDEPLOhy.ppa)            /mDEPLOhy.vDiscTotalOutflows()* 1e3])
OPEX_Electr = pd.Series(data=[(   mDEPLOhy.vTotalMCost() + sum(mDEPLOhy.vUnitOpCost[p,ppa]() for p,ppa in mDEPLOhy.pppa)) /mDEPLOhy.vDiscTotalOutflows()* 1e3])
Revenues    = pd.Series(data=[    mDEPLOhy.vTotalMRevenue()                                                               /mDEPLOhy.vDiscTotalOutflows()*-1e3])
LCOH        = pd.Series(data=[    mDEPLOhy.vTotalSCost()                                                                  /mDEPLOhy.vDiscTotalOutflows()* 1e3])
Output_LCOH = pd.DataFrame({'CAPEX [EUR/kgH2]'          : CAPEX,
                            'OPEXNonEnergy [EUR/kgH2]'  : OPEX_NonEn,
                            'OPEXElectricity [EUR/kgH2]': OPEX_Electr,
                            'Revenues [EUR/kgH2]'       : Revenues,
                            'Global LCOH [EUR/kgH2]'    : LCOH}).to_csv(_path + '/DEPLOhy_Result_00_LCOH_' + CaseName + '.csv', sep=',', header=True)


# EZ Operation

# Global EZ full load hours (FLH) [h] and capacity factor [p.u.]
FLH_step  = {(p,el): (sum(pScenProb[p,sc] * pDuration[n] * mDEPLOhy.vESSCharge[p,sc,n,el]() for sc in PeriodScen[p] for n in mDEPLOhy.n) / (mDEPLOhy.pRatedMaxCharge[el] * mDEPLOhy.vGlobalInvestment()                   )) for p,el in mDEPLOhy.pel}
CapFactor = {(p,el): (sum(pScenProb[p,sc] * pDuration[n] * mDEPLOhy.vESSCharge[p,sc,n,el]() for sc in PeriodScen[p] for n in mDEPLOhy.n) / (mDEPLOhy.pRatedMaxCharge[el] * mDEPLOhy.vGlobalInvestment() * len(mDEPLOhy.nn))) for p,el in mDEPLOhy.pel}
FLH = {}
for el in mDEPLOhy.el:
    for p in mDEPLOhy.p:
        if p == mDEPLOhy.pInitialPeriod[el]:
            FLH[p,el] = pPeriodWeight[p] * FLH_step[p,el]
        elif mDEPLOhy.pInitialPeriod[el] < p < mDEPLOhy.pFinalPeriod[el]:
            FLH[p,el] = FLH[mDEPLOhy.p.prev(p),el] + (pPeriodWeight[p] * FLH_step[p,el])
        else:
            FLH[p,el] = 0.0
FLH       = pd.Series(FLH      , index=pd.Index(mDEPLOhy.pel))
CapFactor = pd.Series(CapFactor, index=pd.Index(mDEPLOhy.pel))

# EZ on, stand-by, and off shares [p.u.]
ShareOn   = pd.Series(data=[(sum(pScenProb[p,sc] * pDuration[n] * mDEPLOhy.vCommitment[p,sc,n,el]() for sc in PeriodScen[p] for n in mDEPLOhy.n) / len(mDEPLOhy.nn)) for p,el in mDEPLOhy.pel], index=pd.Index(mDEPLOhy.pel))
ShareSB   = pd.Series(data=[(sum(pScenProb[p,sc] * pDuration[n] * mDEPLOhy.vStandBy   [p,sc,n,el]() for sc in PeriodScen[p] for n in mDEPLOhy.n) / len(mDEPLOhy.nn)) for p,el in mDEPLOhy.pel], index=pd.Index(mDEPLOhy.pel))
ShareOff  = pd.Series(data=[(sum(pScenProb[p,sc] * pDuration[n] * mDEPLOhy.vOff       [p,sc,n,el]() for sc in PeriodScen[p] for n in mDEPLOhy.n) / len(mDEPLOhy.nn)) for p,el in mDEPLOhy.pel], index=pd.Index(mDEPLOhy.pel))

# EZ consumption range shares  [p.u.]
TimeOn_sc    = pd.Series(data=[ sum(mDEPLOhy.vCommitment[p,sc,n,el]() for n in mDEPLOhy.n for el in mDEPLOhy.el) for p in mDEPLOhy.p for sc in PeriodScen[p]], index=pd.Index(mDEPLOhy.ps))
ZeroShare_sc = pd.Series(data=[(sum(1 for n in mDEPLOhy.n if (                                                                                 mDEPLOhy.vESSCharge2ndBlock[p,sc,n,el]() == 0.0 and mDEPLOhy.vCommitment[p,sc,n,el]() == 1                                )) / TimeOn_sc[p,sc]) for p in mDEPLOhy.p for sc in PeriodScen[p] for el in mDEPLOhy.el], index=pd.Index(mDEPLOhy.psel))
LowShare_sc  = pd.Series(data=[(sum(1 for n in mDEPLOhy.n if (0                                                                              < mDEPLOhy.vESSCharge2ndBlock[p,sc,n,el]() <= 0.3125 * mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,el] * mDEPLOhy.vGlobalInvestment())) / TimeOn_sc[p,sc]) for p in mDEPLOhy.p for sc in PeriodScen[p] for el in mDEPLOhy.el], index=pd.Index(mDEPLOhy.psel))
MedShare_sc  = pd.Series(data=[(sum(1 for n in mDEPLOhy.n if (0.3125 * mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,el] * mDEPLOhy.vGlobalInvestment() < mDEPLOhy.vESSCharge2ndBlock[p,sc,n,el]() <= 0.6875 * mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,el] * mDEPLOhy.vGlobalInvestment())) / TimeOn_sc[p,sc]) for p in mDEPLOhy.p for sc in PeriodScen[p] for el in mDEPLOhy.el], index=pd.Index(mDEPLOhy.psel))
HighShare_sc = pd.Series(data=[(sum(1 for n in mDEPLOhy.n if (0.6875 * mDEPLOhy.pMaxCharge2ndBlock[p,sc,n,el] * mDEPLOhy.vGlobalInvestment() < mDEPLOhy.vESSCharge2ndBlock[p,sc,n,el]()                                                                                  )) / TimeOn_sc[p,sc]) for p in mDEPLOhy.p for sc in PeriodScen[p] for el in mDEPLOhy.el], index=pd.Index(mDEPLOhy.psel))

TimeOn       = pd.Series(data=[ sum(pScenProb[p,sc] * TimeOn_sc   [p,sc   ] for sc in PeriodScen[p]) for p    in mDEPLOhy.p  ], index=pd.Index(mDEPLOhy.p  ))
ZeroShare    = pd.Series(data=[ sum(pScenProb[p,sc] * ZeroShare_sc[p,sc,el] for sc in PeriodScen[p]) for p,el in mDEPLOhy.pel], index=pd.Index(mDEPLOhy.pel))
LowShare     = pd.Series(data=[ sum(pScenProb[p,sc] * LowShare_sc [p,sc,el] for sc in PeriodScen[p]) for p,el in mDEPLOhy.pel], index=pd.Index(mDEPLOhy.pel))
MedShare     = pd.Series(data=[ sum(pScenProb[p,sc] * MedShare_sc [p,sc,el] for sc in PeriodScen[p]) for p,el in mDEPLOhy.pel], index=pd.Index(mDEPLOhy.pel))
HighShare    = pd.Series(data=[ sum(pScenProb[p,sc] * HighShare_sc[p,sc,el] for sc in PeriodScen[p]) for p,el in mDEPLOhy.pel], index=pd.Index(mDEPLOhy.pel))

Output_EZ = pd.DataFrame({'Full load hours [h]': FLH,
                          'CapFactor [p.u.]'   : CapFactor,
                          'On [p.u.]'          : ShareOn,
                          'Stand-by [p.u.]'    : ShareSB,
                          'Off [p.u.]'         : ShareOff,
                          'LowRange [p.u.]'    : ZeroShare + LowShare,
                          'MedRange [p.u.]'    : MedShare,
                          'HighRange [p.u.]'   : HighShare}).to_csv(_path + '/DEPLOhy_Result_01_GlobalEZOperation_' + CaseName + '.csv', sep=',', header=True, index=True)

# Scenario EZ capacity factor [p.u.]
CapFactor_sc = {(p,sc,el): (sum(pDuration[n] * mDEPLOhy.vESSCharge[p,sc,n,el]() for n in mDEPLOhy.n) / (mDEPLOhy.pRatedMaxCharge[el] * mDEPLOhy.vGlobalInvestment() * len(mDEPLOhy.nn))) for p,sc,el in mDEPLOhy.psel}
CapFactor_sc = pd.Series(CapFactor_sc, index=pd.Index(mDEPLOhy.psel))

# EZ on, stand-by, and off shares [p.u.]
ShareOn_sc   = pd.Series(data=[(sum(pDuration[n] * mDEPLOhy.vCommitment[p,sc,n,el]() for n in mDEPLOhy.n) / len(mDEPLOhy.nn)) for p,sc,el in mDEPLOhy.psel], index=pd.Index(mDEPLOhy.psel))
ShareSB_sc   = pd.Series(data=[(sum(pDuration[n] * mDEPLOhy.vStandBy   [p,sc,n,el]() for n in mDEPLOhy.n) / len(mDEPLOhy.nn)) for p,sc,el in mDEPLOhy.psel], index=pd.Index(mDEPLOhy.psel))
ShareOff_sc  = pd.Series(data=[(sum(pDuration[n] * mDEPLOhy.vOff       [p,sc,n,el]() for n in mDEPLOhy.n) / len(mDEPLOhy.nn)) for p,sc,el in mDEPLOhy.psel], index=pd.Index(mDEPLOhy.psel))

Output_EZ_sc = pd.DataFrame({'CapFactor [p.u.]'   : CapFactor_sc,
                             'On [p.u.]'          : ShareOn_sc,
                             'Stand-by [p.u.]'    : ShareSB_sc,
                             'Off [p.u.]'         : ShareOff_sc,
                             'LowRange [p.u.]'    : ZeroShare_sc + LowShare_sc,
                             'MedRange [p.u.]'    : MedShare_sc,
                             'HighRange [p.u.]'   : HighShare_sc}).to_csv(_path + '/DEPLOhy_Result_01_ScenarioEZOperation_' + CaseName + '.csv', sep=',', header=True, index=True)

# Costs summary

# Period LCOH
DiscOutflows_p = pd.Series(data=[((mDEPLOhy.pDiscountFactor[p]   / pPeriodWeight [p]) * sum(mDEPLOhy.vPeriodOutflows[p,sc]() for sc in PeriodScen[p])) / (len(mDEPLOhy.sc)/len(mDEPLOhy.p))  for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))
LCOH_p         = pd.Series(data=[ (mDEPLOhy.vPeriodSCost   [p]() / DiscOutflows_p[p]) * 1e3                                                                                                  for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))

# Period costs results
CAPEX_p        = pd.Series(data=[mDEPLOhy.vPeriodFCost[p]()                                                                          for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))
OPEXNonEl_p    = pd.Series(data=[  sum(mDEPLOhy.vUnitOpCost[p,g  ]() for g   in mDEPLOhy.g if g not in mDEPLOhy.ppa)                 for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))
if len(mDEPLOhy.ppa):
    PPACost_p  = pd.Series(data=[  sum(mDEPLOhy.vUnitOpCost[p,ppa]() for ppa in                        mDEPLOhy.ppa)                 for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))
else:
    PPACost_p  = pd.Series(data=[  0.0                                                                                               for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))
PowerCost_p    = pd.Series(data=[((sum(mDEPLOhy.pDiscountFactor[p]                                  * mDEPLOhy.pPowerTariff      [p,m] * mDEPLOhy.vContractedPower   [p,m]()                                             for m   in mDEPLOhy.m  ) * (1 + pEnergyTax)) * (1 + pVAT)) for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))
PurchaseCost_p = pd.Series(data=[((sum(mDEPLOhy.pDiscountFactor[p] * pScenProb[p,sc] * pDuration[n] * mDEPLOhy.pEnergyCost[p,sc,n,ndb] * mDEPLOhy.vPurchase   [p,sc,n,ndb]() for sc in PeriodScen[p] for n in mDEPLOhy.n for ndb in mDEPLOhy.ndb) * (1 + pEnergyTax)) * (1 + pVAT)) for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))
SalesRevenue_p = pd.Series(data=[mDEPLOhy.vPeriodSaleRevenue[p]()                                                                    for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))
ResRevenue_p   = pd.Series(data=[mDEPLOhy.vPeriodResRevenue[p]()                                                                     for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))
SCost_p        = pd.Series(data=[mDEPLOhy.vPeriodSCost[p]()                                                                          for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))
Outflows_p     = pd.Series(data=[sum(mDEPLOhy.vPeriodOutflows[p,sc]() for sc in PeriodScen[p]) / (len(mDEPLOhy.sc)/len(mDEPLOhy.p))  for p in mDEPLOhy.p], index=pd.Index(mDEPLOhy.p))

Output_data = pd.DataFrame({'CAPEX [MEUR]'                : CAPEX_p,
                            'OPEXNonEnergy [MEUR]'        : OPEXNonEl_p,
                            'PPA cost [MEUR]'             : PPACost_p,
                            'Grid purchase cost [MEUR]'   : PurchaseCost_p,
                            'Contracted power cost [MEUR]': PowerCost_p,
                            'Grid sales revenue [MEUR]'   : SalesRevenue_p,
                            'Reserves revenue [MEUR]'     : ResRevenue_p,
                            'Total system cost [MEUR]'    : SCost_p,
                            'Period LCOH [EUR/kgH2]'      : LCOH_p,
                            'Period outflows [tonH2]'     : Outflows_p}).to_csv(_path + '/DEPLOhy_Result_03_CostsSummary_' + CaseName + '.csv', sep=',', header=True, index=True)


# Unit operation costs results
OutputToFile_UnitOpCost = pd.Series(data=[mDEPLOhy.vUnitOpCost[p,g]()                                              for p,g in mDEPLOhy.pg], index=pd.Index(mDEPLOhy.pg))
OutputToFile_UnitOpCost = OutputToFile_UnitOpCost.fillna(0.0).to_frame(name='Operation Cost [MEUR]').reset_index().rename(columns={'level_0': 'Period', 'level_1': 'Unit'}).to_csv(_path +'/DEPLOhy_Result_04_CostsOperational_'+CaseName+'.csv', sep=',', index=False)


# Unit investment results

if len(mDEPLOhy.gc):

    # Saving investment costs into CSV file
    OutputToFile_ICostg = pd.Series(data=[mDEPLOhy.vUnitICost[p,gc]()                                              for p,gc in mDEPLOhy.pgc], index=pd.Index(mDEPLOhy.pgc))
    OutputToFile_ICostg = OutputToFile_ICostg.fillna(0.0).to_frame(name='Investment Cost [MEUR]'   ).reset_index().rename(columns={'level_0': 'Period', 'level_1': 'Unit'}).to_csv(_path +'/DEPLOhy_Result_05_CostsInvestment_'+CaseName+'.csv', sep=',', index=False)

    # Saving investment decisions into CSV file
    InvDecisions = {}
    for p,g in mDEPLOhy.pg:
        gen_inv  = 0.0
        cons_inv = 0.0
        stor_inv = 0.0
        ppa_inv  = 0.0
        # gc subset
        if g in mDEPLOhy.gc:
            gen_inv  = mDEPLOhy.pRatedMaxPower    [g] * mDEPLOhy.vUnitInvestment[p,g]() * 1e3
            cons_inv = mDEPLOhy.pRatedMaxCharge   [g] * mDEPLOhy.vUnitInvestment[p,g]() * 1e3
        # hc subset
        if g in mDEPLOhy.hc:
            stor_inv = mDEPLOhy.pH2RatedMaxStorage[g] * mDEPLOhy.vUnitInvestment[p,g]() * 1e3
        # ppa subset
        if len(mDEPLOhy.ppa):
            if g in mDEPLOhy.ppa:
                ppa_inv  = mDEPLOhy.pRatedMaxPower[g] * mDEPLOhy.vPPAInvestment [p,g]() * 1e3
            InvDecisions[(p,g)] = {'Generation Investment [MW]': gen_inv, 'Consumption Investment [MW]': cons_inv, 'Storage Investment [kgH2]': stor_inv, 'PPA Decision [MW]': ppa_inv}
        else:
            InvDecisions[(p,g)] = {'Generation Investment [MW]': gen_inv, 'Consumption Investment [MW]': cons_inv, 'Storage Investment [kgH2]': stor_inv}

    OutputToFile_Investment = (pd.DataFrame.from_dict(InvDecisions, orient='index').reset_index().rename(columns={'level_0': 'Period', 'level_1': 'Unit'}))
    OutputToFile_Investment.to_csv(_path + '/DEPLOhy_Result_07_InvestmentDecisions_' + CaseName + '.csv', sep=',', index=False)


# Unit retirement results

if len(mDEPLOhy.gd):

    # Saving retirement costs into CSV file
    OutputToFile_RCostg = pd.Series(data=[mDEPLOhy.vUnitRCost[p,gd]() for p,gd in mDEPLOhy.pgd], index=pd.Index(mDEPLOhy.pgd))
    OutputToFile_RCostg = OutputToFile_RCostg.fillna(0.0).to_frame(name='Retirement Cost [MEUR]').reset_index().rename(columns={'level_0': 'Period', 'level_1': 'Unit'}).to_csv(_path +'/DEPLOhy_Result_06_CostsRetirement_'+CaseName+'.csv', sep=',', index=False)

    # Saving retirement decisions into CSV file
    RetDecisions = {}
    for p,g in mDEPLOhy.pg:
        gen_ret  = 0.0
        cons_ret = 0.0
        stor_ret = 0.0
        # gc subset
        if g in mDEPLOhy.gd:
            gen_ret  = mDEPLOhy.pRatedMaxPower    [g] * mDEPLOhy.vUnitRetirement[p,g]() * 1e3
            cons_ret = mDEPLOhy.pRatedMaxCharge   [g] * mDEPLOhy.vUnitRetirement[p,g]() * 1e3
        # hc subset
        if g in mDEPLOhy.hd:
            stor_ret = mDEPLOhy.pH2RatedMaxStorage[g] * mDEPLOhy.vUnitRetirement[p,g]() * 1e3

        RetDecisions[(p,g)] = {'Generation Retirement [MW]': gen_ret, 'Consumption Retirement [MW]': cons_ret, 'Storage Retirement [kgH2]': stor_ret}

    OutputToFile_Retirement = (pd.DataFrame.from_dict(RetDecisions, orient='index').reset_index().rename(columns={'level_0': 'Period', 'level_1': 'Unit'}))
    OutputToFile_Retirement.to_csv(_path + '/DEPLOhy_Result_08_RetirementDecisions_' + CaseName + '.csv', sep=',', index=False)


# Contracted power decision

OutputToFile_Power = pd.Series(data=[mDEPLOhy.vContractedPower[p,m]() * 1e3 for p,m in mDEPLOhy.pm], index=pd.Index(mDEPLOhy.pm))
OutputToFile_Power = OutputToFile_Power.fillna(0.0).to_frame(name='Contracted Power [MW]').reset_index().rename(columns={'level_0': 'Period', 'level_1': 'MarketPeriod'}).to_csv(_path +'/DEPLOhy_Result_09_ContractedPower_'+CaseName+'.csv', sep=',', index=False)


# Network operation results

OutputResults = pd.Series(data=[mDEPLOhy.vFlow[p,sc,n,ni,nf,cc]() for p,sc,n,ni,nf,cc in mDEPLOhy.psnla], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnla))
OutputResults *= 1e3
OutputResults.index.names = ['Period', 'Scenario', 'LoadLevel', 'InitialNode', 'FinalNode', 'Circuit']
OutputResults = pd.pivot_table(OutputResults.to_frame(name='MW'), values='MW', index=['Period', 'Scenario', 'LoadLevel'], columns=['InitialNode', 'FinalNode', 'Circuit'], fill_value=0.0).rename_axis([None, None, None], axis=1)
OutputResults.reset_index().to_csv(_path+'/DEPLOhy_Result_11_NetworkFlowPerNode_'+CaseName+'.csv', index=False, sep=',')

OutputResults = pd.Series(data=[(sum((mDEPLOhy.vFlow[p,sc,n,ni,nf,cc]()*pDuration[n]) for n in mDEPLOhy.n) * pPeriodWeight[p]) for p,sc,ni,nf,cc in mDEPLOhy.psla if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psla))
OutputResults *= 1e3
OutputResults.index.names = ['Period', 'Scenario', 'InitialNode', 'FinalNode', 'Circuit']
OutputResults = pd.pivot_table(OutputResults.to_frame(name='MWh'), values='MWh', index=['Period', 'Scenario'], columns=['InitialNode', 'FinalNode', 'Circuit'], fill_value=0.0).rename_axis([None, None, None], axis=1)
OutputResults.reset_index().to_csv(_path+'/DEPLOhy_Result_12_NetworkBalance_'+CaseName+'.csv', index=False, sep=',')


# Hourly operation results

OutputCommitment = pd.Series(data=[mDEPLOhy.vCommitment[p,sc,n,el]()                               for p,sc,n,el in mDEPLOhy.psnel if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnel))
OutputCommitment = (OutputCommitment.to_frame(name='p.u.').reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='p.u.').rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Commitment {x}'))

OutputStandBy    = pd.Series(data=[mDEPLOhy.vStandBy   [p,sc,n,el]()                               for p,sc,n,el in mDEPLOhy.psnel if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnel))
OutputStandBy    = (OutputStandBy.to_frame(name='p.u.'   ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='p.u.').rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'StandBy {x}'))

OutputOff        = pd.Series(data=[mDEPLOhy.vOff       [p,sc,n,el]()                               for p,sc,n,el in mDEPLOhy.psnel if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnel))
OutputOff        = (OutputOff.to_frame(name='p.u.'       ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='p.u.').rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Off {x}'))

OutputStartUp    = pd.Series(data=[mDEPLOhy.vStartUp   [p,sc,n,el]()                               for p,sc,n,el in mDEPLOhy.psnel if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnel))
OutputStartUp    = (OutputStartUp.to_frame(name='p.u.'   ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='p.u.').rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'StartUp {x}'))

OutputShutDown   = pd.Series(data=[mDEPLOhy.vShutDown  [p,sc,n,el]()                               for p,sc,n,el in mDEPLOhy.psnel if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnel))
OutputShutDown   = (OutputShutDown.to_frame(name='p.u.'  ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='p.u.').rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ShutDown {x}'))

OutputH2Inflows  = pd.Series(data=[mDEPLOhy.vH2Production[p,sc,n,el]()*1e3                         for p,sc,n,el in mDEPLOhy.psnel if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnel))
OutputH2Inflows  = (OutputH2Inflows.to_frame(name='kg/h' ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='kg/h').rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'H2Inflows {x} [kg/h]'))

OutputH2Outflows = pd.Series(data=[mDEPLOhy.vH2Outflows[p,sc,n,hs]()*1e3                           for p,sc,n,hs in mDEPLOhy.psnhs if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnhs))
OutputH2Outflows = (OutputH2Outflows.to_frame(name='kg/h').reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='kg/h').rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'H2Outflows {x} [kg/h]'))

OutputGenOut     = pd.Series(data=[mDEPLOhy.vOutput[p,sc,n,gp]()*1e3                               for p,sc,n,gp in mDEPLOhy.psngp if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psngp))
OutputGenOut     = (OutputGenOut.to_frame(name='MW'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'GenOutput {x} [MW]'))

if pIndReserves == 1:
    OutputResUpC = pd.Series(data=[mDEPLOhy.vReserveUpC[p,sc,n,nr]()*1e3                           for p,sc,n,nr in mDEPLOhy.psnnr if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnnr))
    OutputResUpC = (OutputResUpC.to_frame(name='MW'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ResUpC {x} [MW]'))

    OutputResDwC = pd.Series(data=[mDEPLOhy.vReserveDwC[p,sc,n,nr]()*1e3                           for p,sc,n,nr in mDEPLOhy.psnnr if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnnr))
    OutputResDwC = (OutputResDwC.to_frame(name='MW'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ResDwC {x} [MW]'))

    OutputEnUpC  = pd.Series(data=[mDEPLOhy.vReserveUpC[p,sc,n,nr]()*1e3*mDEPLOhy.pActivationUp[p,sc,n]*pDuration[n] for p,sc,n,nr in mDEPLOhy.psnnr if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnnr))
    OutputEnUpC  = (OutputEnUpC.to_frame(name='MWh'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MWh' ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'EnergyUpC {x} [MWh]'))

    OutputEnDwC  = pd.Series(data=[mDEPLOhy.vReserveDwC[p,sc,n,nr]()*1e3*mDEPLOhy.pActivationDw[p,sc,n]*pDuration[n] for p,sc,n,nr in mDEPLOhy.psnnr if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnnr))
    OutputEnDwC  = (OutputEnDwC.to_frame(name='MWh'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MWh' ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'EnergyDwC {x} [MWh]'))

    OutputActivC = pd.Series(data=[mDEPLOhy.vActivationC[p,sc,n,nr]()                              for p,sc,n,nr in mDEPLOhy.psnnr if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnnr))
    OutputActivC = (OutputActivC.to_frame(name='p.u.'    ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='p.u.').rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ActivationC {x}'))

if pIndBinInvest == 2:
    OutputCurtailm  = pd.Series(data=[(pMaxPower[re][p,sc,n]                                   -mDEPLOhy.vOutput[p,sc,n,re]())*1e3 for p,sc,n,re in mDEPLOhy.psnre if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnre))
    OutputCurtailm  = (OutputCurtailm.to_frame(name='MW' ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Curtailment {x} [MW]'))
else:
    OutputCurtailm  = pd.Series(data=[((pMaxPower[re][p,sc,n]*mDEPLOhy.vUnitInvestment[p,re]())-mDEPLOhy.vOutput[p,sc,n,re]())*1e3 for p,sc,n,re in mDEPLOhy.psnre if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnre))
    OutputCurtailm  = (OutputCurtailm.to_frame(name='MW' ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Curtailment {x} [MW]'))

if len(mDEPLOhy.ppa):
    OutputPPACurt   = pd.Series(data=[mDEPLOhy.vCurtailment[p,sc,n,ppa]()*1e3                                        for p,sc,n,ppa in mDEPLOhy.psnppa if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnppa))
    OutputPPACurt   = (OutputPPACurt.to_frame(name='MW'  ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PPACurtailment {x} [MW]'))

    OutputPPACons   = pd.Series(data=[(mDEPLOhy.vOutput    [p,sc,n,ppa]() - mDEPLOhy.vCurtailment[p,sc,n,ppa]())*1e3 for p,sc,n,ppa in mDEPLOhy.psnppa if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnppa))
    OutputPPACons   = (OutputPPACons.to_frame(name='MW'  ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PPAConsumption {x} [MW]'))


# ESS storage results

OutputCharge     = pd.Series(data=[mDEPLOhy.vESSCharge[p,sc,n,es]()*1e3                            for p,sc,n,es in mDEPLOhy.psnes if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnes))
OutputCharge     = (OutputCharge.to_frame(name='MW'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ChargeOutput {x} [MW]'))

OutputInventory  = pd.Series(data=[mDEPLOhy.vESSInventory[p,sc,n,su]()                             for p,sc,n,su in mDEPLOhy.psnsu if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnsu))
OutputInventory  = (OutputInventory.to_frame(name='GWh'  ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='GWh' ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Inventory {x} [GWh]'))

OutputSpillage   = pd.Series(data=[mDEPLOhy.vESSSpillage[p,sc,n,su]()                              for p,sc,n,su in mDEPLOhy.psnsu if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnsu))
OutputSpillage   = (OutputSpillage.to_frame(name='GWh'   ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='GWh' ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Spillage {x} [GWh]'))

OutputChargeInd  = pd.Series(data=[mDEPLOhy.vChargeInd[p,sc,n,su]()                                for p,sc,n,su in mDEPLOhy.psnsu if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnsu))
OutputChargeInd  = (OutputChargeInd.to_frame(name='p.u.' ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='p.u.').rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ChargeInd {x}'))

if pIndReserves == 1:
    OutputResUpD = pd.Series(data=[mDEPLOhy.vReserveUpD[p,sc,n,su]()*1e3                           for p,sc,n,su in mDEPLOhy.psnsu if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnsu))
    OutputResUpD = (OutputResUpD.to_frame(name='MW'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ResUpD {x} [MW]'))

    OutputResDwD = pd.Series(data=[mDEPLOhy.vReserveDwD[p,sc,n,su]()*1e3                           for p,sc,n,su in mDEPLOhy.psnsu if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnsu))
    OutputResDwD = (OutputResDwD.to_frame(name='MW'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ResDwD {x} [MW]'))

    OutputEnUpD  = pd.Series(data=[mDEPLOhy.vReserveUpD[p,sc,n,su]()*1e3*mDEPLOhy.pActivationUp[p,sc,n]*pDuration[n] for p,sc,n,su in mDEPLOhy.psnsu if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnsu))
    OutputEnUpD  = (OutputEnUpD.to_frame(name='MWh'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MWh' ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'EnergyUpD {x} [MWh]'))

    OutputEnDwD  = pd.Series(data=[mDEPLOhy.vReserveDwD[p,sc,n,su]()*1e3*mDEPLOhy.pActivationDw[p,sc,n]*pDuration[n] for p,sc,n,su in mDEPLOhy.psnsu if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnsu))
    OutputEnDwD  = (OutputEnDwD.to_frame(name='MWh'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='MWh' ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'EnergyDwD {x} [MWh]'))

    OutputActivD = pd.Series(data=[mDEPLOhy.vActivationD[p,sc,n,su]()                              for p,sc,n,su in mDEPLOhy.psnsu if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnsu))
    OutputActivD = (OutputActivD.to_frame(name='p.u.'    ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='p.u.').rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ActivationD {x}'))


# H2 Storage results

OutputH2Invent  = pd.Series(data=[mDEPLOhy.vH2Inventory[p,sc,n,hs]()*1e3                           for p,sc,n,hs in mDEPLOhy.psnhs if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnhs))
OutputH2Invent  = (OutputH2Invent.to_frame(name='kg'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='kg'   ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis(None,  axis=1).rename(columns=lambda x: f'H2Inventory {x} [kg]'))

OutputH2Spill   = pd.Series(data=[mDEPLOhy.vH2Spillage[p,sc,n,hs]()*1e3                            for p,sc,n,hs in mDEPLOhy.psnhs if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnhs))
OutputH2Spill   = (OutputH2Spill.to_frame(name='kg'      ).reset_index().pivot_table(index=['level_0','level_1','level_2'], columns='level_3', values='kg'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'H2Spillage {x} [kg]'))


# Power exchange results

OutputPurchase = pd.Series(data=[mDEPLOhy.vPurchase[p,sc,n,ndb]()*1e3                              for p,sc,n,ndb in mDEPLOhy.psnnb if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnnb))
OutputPurchase = OutputPurchase.reset_index()
OutputPurchase['MW'] = OutputPurchase[0]
OutputPurchase = OutputPurchase.pivot_table(index=                                         ['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename(columns=lambda x: f'Purchase {x} [MW]')

OutputSale     = pd.Series(data=[mDEPLOhy.vSale    [p,sc,n,nds]()*1e3                              for p,sc,n,nds in mDEPLOhy.psnns if sc in PeriodScen[p]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psnns))
OutputSale     = OutputSale.reset_index()
OutputSale['MW'] = OutputSale[0]
OutputSale     = OutputSale.pivot_table(index=                                             ['level_0','level_1','level_2'], columns='level_3', values='MW'  ).rename_axis(['Period','Scenario','LoadLevel'], axis=0).rename(columns=lambda x: f'Sale {x} [MW]')


# GHG emissions results [gCO2eq/MJ_h2]

OutputEmissions  = pd.Series(data=[(num / den if den > 0 else 0.0) for p,sc,n in mDEPLOhy.psn if sc in PeriodScen[p] for num,den in [(sum(mDEPLOhy.vPurchase[p,sc,n,ndb]()*3600*pECI[p,sc,n,ndb] for ndb in mDEPLOhy.ndb), sum(mDEPLOhy.vH2Production[p,sc,n,el]()*120 for el in mDEPLOhy.el))]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psn))
OutputEmissions.index.names = ['Period', 'Scenario', 'LoadLevel']
OutputEmissions  = pd.pivot_table(OutputEmissions.to_frame(name='gCO2/MJh2'), values='gCO2/MJh2', index=['Period', 'Scenario', 'LoadLevel'], fill_value=0.0).rename_axis(None, axis=1)
OutputEmissions.reset_index()

HourlyEmissions  = pd.Series(data=[(num / den if den > 0 else 0.0) for p,sc,n in mDEPLOhy.psn if sc in PeriodScen[p] for num,den in [(sum(mDEPLOhy.vPurchase[p,sc,n,ndb]()*3600*pECI[p,sc,n,ndb] for ndb in mDEPLOhy.ndb), sum(mDEPLOhy.vH2Production[p,sc,n,el]()*120 for el in mDEPLOhy.el))]], index=pd.MultiIndex.from_tuples(mDEPLOhy.psn))
HourlyEmissions.index.set_names(['Period', 'Scenario', 'Time'], inplace=True)
pComparator      = (1 - pSavingTarget) * pFossilComparator * 1.0001
InfractionHours  = (((HourlyEmissions > pComparator).astype(float) * pTimeStep * HourlyEmissions.index.get_level_values('Period').map(pPeriodWeight)).groupby(level=0).sum().to_frame(name='Infraction hours [h]'))
AverageEmissions = HourlyEmissions.groupby(level=0).mean().to_frame(name='Average emissions [gCO2eq/MJh2]')
pd.concat([InfractionHours, AverageEmissions], axis=1).reset_index().rename(columns={'level_0': 'Period'}).to_csv(_path + '/DEPLOhy_Result_10_EmissionsSummary_' + CaseName + '.csv', index=False)


# Creating a dataframe with all operation results

if len(mDEPLOhy.ppa):
    if pIndReserves == 1:
        dfs = [OutputCommitment, OutputStandBy, OutputOff, OutputStartUp, OutputShutDown,
               OutputH2Inflows, OutputEmissions, OutputH2Outflows, OutputH2Invent, OutputH2Spill,
               OutputGenOut, OutputCurtailm,
               OutputPPACons, OutputPPACurt,
               OutputChargeInd, OutputCharge, OutputInventory, OutputSpillage,
               OutputResUpD, OutputResDwD, OutputActivD, OutputEnUpD, OutputEnDwD, OutputResUpC, OutputResDwC, OutputActivC, OutputEnUpC, OutputEnDwC,
               OutputPurchase, OutputSale]
    else:
        dfs = [OutputCommitment, OutputStandBy, OutputOff, OutputStartUp, OutputShutDown,
               OutputH2Inflows, OutputEmissions, OutputH2Outflows, OutputH2Invent, OutputH2Spill,
               OutputGenOut, OutputCurtailm,
               OutputPPACons, OutputPPACurt,
               OutputChargeInd, OutputCharge, OutputInventory, OutputSpillage,
               OutputPurchase, OutputSale]
else:
    if pIndReserves == 1:
        dfs = [OutputCommitment, OutputStandBy, OutputOff, OutputStartUp, OutputShutDown,
               OutputH2Inflows, OutputEmissions, OutputH2Outflows, OutputH2Invent, OutputH2Spill,
               OutputGenOut, OutputCurtailm,
               OutputChargeInd, OutputCharge, OutputInventory, OutputSpillage,
               OutputResUpD, OutputResDwD, OutputActivD, OutputEnUpD, OutputEnDwD, OutputResUpC, OutputResDwC, OutputActivC, OutputEnUpC, OutputEnDwC,
               OutputPurchase, OutputSale]
    else:
        dfs = [OutputCommitment, OutputStandBy, OutputOff, OutputStartUp, OutputShutDown,
               OutputH2Inflows, OutputEmissions, OutputH2Outflows, OutputH2Invent, OutputH2Spill,
               OutputGenOut, OutputCurtailm,
               OutputChargeInd, OutputCharge, OutputInventory, OutputSpillage,
               OutputPurchase, OutputSale]

dfResult = pd.concat(dfs, axis=1, join='outer').to_csv(_path + '/DEPLOhy_Result_02_OperationHourly_' + CaseName + '.csv', sep=',', header=True, index=True)

# Reading data from CSV file
dfResult = pd.read_csv(_path+'/DEPLOhy_Result_02_OperationHourly_' + CaseName + '.csv', index_col=[0])

# Converting load levels to a list
lLoadLevel = dfResult.iloc[:,1].values.tolist()

# Adding year to every value
lYear = dfResult.index.tolist()
lLL = [f"{year}-{load}" for year, load in zip(lYear, lLoadLevel)]

# Converting series to datetime object
datetime_series = pd.to_datetime(pd.Series(lLL), errors='coerce', utc=True)
datetime_series = datetime_series.dt.tz_convert('Europe/Madrid')
datetime_series = datetime_series.dropna()

# Extracting individual components
year  = datetime_series.dt.year
month = datetime_series.dt.month
day   = datetime_series.dt.day
hour  = datetime_series.dt.hour

# Creating dataframe with extracted components
dfDatetime = pd.DataFrame({'Year': year, 'Month': month, 'Day': day, 'Hour': hour}, index=datetime_series.index)

dfResult = dfResult.reset_index(drop=True)
dfResult = pd.concat([dfDatetime, dfResult], axis=1)
dfResult.to_csv(_path+'/DEPLOhy_Result_02_OperationHourly_'+ CaseName +'.csv', sep=',', index=False)

WritingResultsTime = time.time() - StartTime
StartTime          = time.time()
print('Writing output results          ... ', round(WritingResultsTime), 's')
print('Total time                      ... ', round((ReadingDataTime + SettingUpDataTime + GeneratingOFTime + GeneratingIRCTime + GeneratingEGCTime + GeneratingSOTime + GeneratingORTime + GeneratingCCTime + GeneratingLZTime + GeneratingECTime + SolvingTime + WritingResultsTime)/60, ndigits=2), 'min')
print('\n #### Non-commercial use only #### \n')