def load_query (multiple_predict, project_name, reg_datekey_str, predict_min_date,predict_max_date):
    if  multiple_predict == False:
        query = f'''
    -- 통합
    with sample as(
      select A.reg_datekey, A.datekey, reg_uu, date_diff(A.datekey, A.reg_datekey, DAY) as reg_diff, DAU
      , coalesce(DAU , 0)/ reg_uu as retention
      , coalesce(daily_revenue , 0)/ DAU as ARPU
      , min(A.reg_datekey) over(order by A.reg_datekey) as launching_date
      from(
        select reg_datekey, datekey, count(distinct nid) as DAU, sum(daily_revenue) as daily_revenue
        from {project_name.lower()}_dw.f_user_map
        where login_flag = 1
        and datekey>= '{reg_datekey_str}' and datekey<= date_add('{reg_datekey_str}', interval 30 day)
        and reg_datekey = '{reg_datekey_str}'
        group by reg_datekey, datekey
      )as A
      INNER JOIN(
        -- 인원수 확인
        select reg_datekey, count(distinct nid) as reg_uu
        from {project_name.lower()}_dw.f_user_map
        where nru = 1
        and datekey>= '{reg_datekey_str}' and datekey<= date_add('{reg_datekey_str}', interval 30 day)
        and reg_datekey = '{reg_datekey_str}'
        group by reg_datekey
      )as B
      ON A.reg_datekey = B.reg_datekey
    )
    , sample2 as (
      select reg_datekey, datekey, reg_uu, reg_diff--, date_diff(datekey, launching_date, DAY) as reg_diff
      , DAU, retention, ARPU
      , sum( retention * ARPU ) over(partition by reg_datekey order by datekey rows between unbounded preceding and current row)as LTV
      from sample
    )
    
    select A.datekey, A.reg_datekey, B.daily_revenue
    , retention, ARPU, DAU, reg_diff--, reg_diff
    from (
      select *
      from sample2
      where reg_diff <= 30
    )as A
    INNER JOIN (
      select reg_datekey,datekey, sum(daily_revenue) as daily_revenue
      from {project_name.lower()}_dw.f_user_map
      where datekey>= '{reg_datekey_str}' and datekey<= date_add('{reg_datekey_str}', interval 30 day)
      and reg_datekey = '{reg_datekey_str}'
      group by reg_datekey,datekey
    )as B
    ON A.datekey = B.datekey and A.reg_datekey = B.reg_datekey
    order by reg_datekey, datekey'''
    else:
        min_reg_datekey =  min(reg_datekey_str)#.strftime('%Y-%m-%d')
        max_reg_datekey = max(reg_datekey_str)#.strftime('%Y-%m-%d')

        # reg_datekey_list = [x.strftime('%Y-%m-%d') for x in reg_datekey_str ]
        reg_datekey_list = "', '".join(reg_datekey_str)

        query = f'''
    with sample as(
      select A.reg_datekey, A.datekey, reg_uu, date_diff(A.datekey, A.reg_datekey, DAY) as reg_diff, DAU
      , coalesce(DAU , 0)/ reg_uu as retention
      , coalesce(daily_revenue , 0)/ DAU as ARPU
      --, min(A.reg_datekey) over(order by A.reg_datekey) as launching_date
      from(
        select reg_datekey, datekey, count(distinct nid) as DAU, sum(daily_revenue) as daily_revenue
        from {project_name.lower()}_dw.f_user_map
        where login_flag = 1
        and datekey>= '{predict_min_date}' and datekey<= '{predict_max_date}'
        and reg_datekey in ('{reg_datekey_list}')
        group by reg_datekey, datekey
      )as A
      INNER JOIN(
        -- 인원수 확인
        select reg_datekey, count(distinct nid) as reg_uu
        from {project_name.lower()}_dw.f_user_map
        where nru = 1
        and datekey>= '{predict_min_date}' and datekey<= '{predict_max_date}'
        and reg_datekey in ('{reg_datekey_list}')
        group by reg_datekey
      )as B
      ON A.reg_datekey = B.reg_datekey
    )
    , sample2 as (
      select reg_datekey, datekey, reg_uu, reg_diff, DAU, retention, ARPU
      from sample
    )
    
    select A.datekey, A.reg_datekey, B.daily_revenue
    , retention, ARPU, DAU, reg_diff
    from (
      select *
      from sample2
      --where reg_diff <= 30 -- 실제값과 비교할 최대 데이터 수
    )as A
    INNER JOIN (
      select reg_datekey,datekey, sum(daily_revenue) as daily_revenue
      from {project_name.lower()}_dw.f_user_map
      where datekey>= '{predict_min_date}' and datekey<= '{predict_max_date}'
        and reg_datekey in ('{reg_datekey_list}')
      group by reg_datekey,datekey
    )as B
    ON A.datekey = B.datekey and A.reg_datekey = B.reg_datekey
    order by reg_datekey, datekey
    '''

    return query
