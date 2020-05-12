def load_query (project_name, reg_datekey_str):
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

    return query