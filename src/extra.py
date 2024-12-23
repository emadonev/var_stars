def blazhko_determine(df0):
    # Step 01 - filter unwanted df_cs
    # ===========

    df_c = df0[
    ((df0['Lampl'] < 2)|
    (df0['Zampl'] < 2)) &
    (df0['Ampl_diff'] < 2) &
    ((df0['Plinear'] < 4) | 
    (df0['Pztf'] < 4)) &
    (df0['NdataLINEAR'] >= 150) &
    (df0['NdataZTF'] >= 150) &
    (df0['Pratio'] > 0.8) &
    (df0['Pratio'] < 1.2)]

    
    # STEP 02 - filter blazhko candidates
    snr = 0.15
    snr1 = 0.10
    dPmin = 0.01

    LINEAR_pd_period = (
        (np.abs(df_c['Plinear'] - 0.5) > dPmin) &
        (np.abs(df_c['Plinear'] - 1.0) > dPmin) &
        (np.abs(df_c['Plinear'] - 2.0) > dPmin) &
        (df_c['BlazhkoPeriodL'] > 35) &
        (df_c['BlazhkoPeriodL'] < 325) &
        (df_c['BpowerRatioL'] > snr) &
        (df_c['BsignificanceL'] > 5)
        )

    ZTF_pd_period = (
            (np.abs(df_c['Pztf'] - 0.5) > dPmin) &
            (np.abs(df_c['Pztf'] - 1.0) > dPmin) &
            (np.abs(df_c['Pztf'] - 2.0) > dPmin) &
            (df_c['BlazhkoPeriodZ'] > 35) &
            (df_c['BlazhkoPeriodZ'] < 325) &
            (df_c['BpowerRatioZ'] > snr) &
            (df_c['BsignificanceZ'] > 5)
        )

    BOTH_pd_period = (
            (np.abs(df_c['Plinear'] - 0.5) > dPmin) &
            (np.abs(df_c['Plinear'] - 1.0) > dPmin) &
            (np.abs(df_c['Plinear'] - 2.0) > dPmin) &
            (df_c['BlazhkoPeriodL'] > 35) &
            (df_c['BlazhkoPeriodL'] < 325) &
            (df_c['BpowerRatioL'] > snr1) &
            (df_c['BsignificanceL'] > 5)&
            (np.abs(df_c['Pztf'] - 0.5) > dPmin) &
            (np.abs(df_c['Pztf'] - 1.0) > dPmin) &
            (np.abs(df_c['Pztf'] - 2.0) > dPmin) &
            (df_c['BlazhkoPeriodZ'] > 35) &
            (df_c['BlazhkoPeriodZ'] < 325) &
            (df_c['BpowerRatioZ'] > snr1) &
            (df_c['BsignificanceZ'] > 5)
        )        
    
    df_c = df_c.copy()

    df_c.loc[BOTH_pd_period, 'IndicatorType'] = 'LZ'
    df_c.loc[LINEAR_pd_period, 'IndicatorType'] = 'L'
    df_c.loc[ZTF_pd_period, 'IndicatorType'] = 'Z'

    # STEP 3: separate periodogram stars from the rest
    BE = df_c[df_c['IndicatorType'] != '-']
    BE_non = df_c[df_c['IndicatorType'] == '-']

    
    # STEP 4: chi2 and period masking
    
    # CHI^2 MASKING
    # LZ3
    BE_non['BE_score'] = 0

    LZ3 = (((BE_non['L_chi2dofR'] > 2.0)&
        (BE_non['L_chi2dofR'] < 3.0)&
        (BE_non['Zchi2dofR'] > 2.0)&
        (BE_non['Zchi2dofR'] < 3.0))|
        
        ((BE_non['L_chi2dofR'] > 2.0)&
        (BE_non['L_chi2dofR'] < 3.0)&
        (BE_non['Zchi2dofR'] < 3.0))|
        
        ((BE_non['L_chi2dofR'] < 3.0)&
        (BE_non['Zchi2dofR'] > 2.0)&
        (BE_non['Zchi2dofR'] < 3.0))) # a mask that will check for every df_c if this is true or not
    
    BE_non.loc[LZ3, 'BE_score'] += 3
    BE_non.loc[LZ3, 'ChiType'] = 'LZ3'

    # LZ4 masking
    LZ4 = (
        ((BE_non['L_chi2dofR'] > 3.0)&
            (BE_non['L_chi2dofR'] < 5.0)&
            (BE_non['L_chi2dofR'] < 3.0))|

        ((BE_non['L_chi2dofR'] < 3.0)&
            (BE_non['Zchi2dofR'] > 3.0)&
            (BE_non['L_chi2dofR'] < 5.0)))
    
    BE_non.loc[LZ4, 'BE_score'] += 4
    BE_non.loc[LZ4, 'ChiType'] = 'LZ4'

    # LZ5 masking
    LZ5 = (
        ((BE_non['L_chi2dofR'] > 5.0)&
            (BE_non['Zchi2dofR'] > 5.0))|

        ((BE_non['L_chi2dofR'] > 3.0)&
            (BE_non['L_chi2dofR'] < 5.0)&
            (BE_non['Zchi2dofR'] > 3.0)&
            (BE_non['Zchi2dofR'] < 5.0))|

        ((BE_non['L_chi2dofR'] > 5.0)&
            (BE_non['Zchi2dofR'] < 5.0))|

        ((BE_non['L_chi2dofR'] < 5.0)&
            (BE_non['Zchi2dofR'] > 5.0)))
    
    BE_non.loc[LZ5, 'BE_score'] += 5
    BE_non.loc[LZ5, 'ChiType'] = 'LZ5'
    
    # PERIOD AND AMPLITUDE MASKING
    amp1 = (
        (BE_non['Ampl_diff'] > 0.05)&
        (BE_non['Ampl_diff'] < 0.15)
    )
    BE_non.loc[amp1, 'BE_score'] += 1

    amp2 = (
        (BE_non['Ampl_diff'] >= 0.15)&
        (BE_non['Ampl_diff'] < 2.0)
    )
    BE_non.loc[amp2, 'BE_score'] += 2

    p1 = (
        (BE_non['dP'] > 0.00002)&
        (BE_non['dP'] < 0.00005)
    )
    BE_non.loc[p1, 'BE_score'] += 1

    p2 = ((BE_non['dP'] >= 0.00005))
    BE_non.loc[p2, 'BE_score'] += 2

    # FINAL STEP: putting it all together
    BE_cand_other = BE_non[BE_non['BE_score'] > 4]

    b_candidates = pd.concat([BE, BE_cand_other], ignore_index=True)
    #'''
    return b_candidates