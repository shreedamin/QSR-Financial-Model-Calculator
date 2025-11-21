
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- Helper functions -----------------


def build_orders_with_ramp_and_postgrowth(
    start_orders: float,
    target_orders: float,
    ramp_months: int,
    total_months: int,
    post_ramp_growth_per_month: float,
    max_daily_orders: float,
):
    """
    Linear ramp from start_orders to target_orders over ramp_months,
    then continue growing by post_ramp_growth_per_month each month
    until hitting max_daily_orders.
    """
    months = np.arange(1, total_months + 1)
    orders = np.zeros_like(months, dtype=float)

    if ramp_months <= 1:
        # Immediate jump to target, then post-ramp growth
        for i, m in enumerate(months, start=1):
            if m == 1:
                ords = target_orders
            else:
                ords = min(target_orders + post_ramp_growth_per_month * (m - 1), max_daily_orders)
            orders[i - 1] = ords
        return orders

    # Ramp period
    for i, m in enumerate(months, start=1):
        if m <= ramp_months:
            t = (m - 1) / (ramp_months - 1)
            orders[i - 1] = start_orders + t * (target_orders - start_orders)
        else:
            # Post-ramp growth
            extra_months = m - ramp_months
            ords = target_orders + post_ramp_growth_per_month * extra_months
            orders[i - 1] = min(ords, max_daily_orders)

    return orders


def compute_5yr_projection(
    start_orders: float,
    target_orders: float,
    ramp_months: int,
    days_per_month: float,
    ticket: float,
    ticket_infl: float,
    max_ticket_price: float,
    cogs_pct: float,
    labor_pct: float,
    other_pct: float,
    rent_month_y1: float,
    rent_infl: float,
    cogs_infl: float,
    labor_infl: float,
    other_infl: float,
    # labor floor params
    min_employees: int,
    hourly_rate: float,
    labor_burden_mult: float,
    hours_per_employee_month: float,
    investment: float,
    # post-ramp growth
    post_ramp_growth_per_month: float,
    max_daily_orders: float,
    # misc expenses
    misc_expense_pct: float = 0.0,
    misc_expense_fixed: float = 0.0,
    misc_expense_infl: float = 0.0,
    total_months: int = 120,
):
    """
    10-year (120 month) projection with:
      • daily orders ramping and then continuing to grow
      • annual inflation on ticket price, cost percentages and rent
      • minimum staffing floor for labor cost
    All values returned are MONTHLY.
    """
    orders = build_orders_with_ramp_and_postgrowth(
        start_orders,
        target_orders,
        ramp_months,
        total_months,
        post_ramp_growth_per_month,
        max_daily_orders,
    )

    months = np.arange(1, total_months + 1)
    years = (months - 1) // 12 + 1

    revenues = []
    cogs_list = []
    labor_list = []
    labor_target_list = []  # Percentage-based labor (target)
    labor_actual_list = []  # Minimum staffing floor (actual)
    other_list = []
    rent_list = []
    profit_list = []
    gross_list = []
    occ_pct_list = []
    labor_pct_actual_list = []

    base_min_labor = min_employees * hourly_rate * labor_burden_mult * hours_per_employee_month

    for m, y, ords in zip(months, years, orders):
        year_idx = int(y - 1)  # 0 for Year 1, 1 for Year 2, etc.

        # Ticket price with cap
        ticket_eff = ticket * (1 + ticket_infl) ** year_idx
        ticket_eff = min(ticket_eff, max_ticket_price)

        revenue = ords * days_per_month * ticket_eff

        # Inflation factors
        c_factor = (1 + cogs_infl) ** year_idx
        l_factor = (1 + labor_infl) ** year_idx
        o_factor = (1 + other_infl) ** year_idx
        rent_factor = (1 + rent_infl) ** year_idx

        # Percentage-based costs
        cogs = revenue * cogs_pct * c_factor
        other = revenue * other_pct * o_factor
        
        # Misc expenses (either percentage-based or fixed, with inflation)
        misc_factor = (1 + misc_expense_infl) ** year_idx
        if misc_expense_pct > 0:
            misc_expenses = revenue * misc_expense_pct * misc_factor
        else:
            misc_expenses = misc_expense_fixed * misc_factor
        
        # Add misc expenses to other OpEx
        other = other + misc_expenses

        # Labor: max(percentage-based, staffing floor with inflation)
        pct_labor_cost = revenue * labor_pct * l_factor  # Target spend
        min_labor_cost_y = base_min_labor * (1 + labor_infl) ** year_idx  # Actual spend (staffing floor)
        labor = max(pct_labor_cost, min_labor_cost_y)  # Used for profit calculation

        # Rent with inflation
        rent = rent_month_y1 * rent_factor

        profit = revenue - (cogs + labor + other + rent)
        gross = revenue - cogs
        occ_pct = rent / revenue if revenue > 0 else np.nan
        labor_pct_actual = labor / revenue if revenue > 0 else np.nan

        revenues.append(revenue)
        cogs_list.append(cogs)
        labor_list.append(labor)
        labor_target_list.append(pct_labor_cost)
        labor_actual_list.append(min_labor_cost_y)
        other_list.append(other)
        rent_list.append(rent)
        profit_list.append(profit)
        gross_list.append(gross)
        occ_pct_list.append(occ_pct)
        labor_pct_actual_list.append(labor_pct_actual)

    revenues = np.array(revenues)
    profit_list = np.array(profit_list)
    gross_list = np.array(gross_list)
    cumulative = profit_list.cumsum()

    # Find payback month (first month cumulative >= investment)
    payback_month = None
    for i, val in enumerate(cumulative):
        if val >= investment:
            payback_month = int(i + 1)
            break

    df = pd.DataFrame(
        {
            "Month": months,
            "Year": years,
            "Daily Orders": orders,
            "Monthly Revenue": revenues,
            "COGS (Monthly)": cogs_list,
            "Labor (Monthly)": labor_list,
            "Labor (Target)": labor_target_list,
            "Labor (Actual)": labor_actual_list,
            "Other OpEx (Monthly)": other_list,
            "Rent (Monthly)": rent_list,
            "Occupancy % (Rent/Revenue)": occ_pct_list,
            "Labor % (Actual)": labor_pct_actual_list,
            "Monthly Profit": profit_list,
            "Monthly Gross Profit": gross_list,
            "Cumulative Profit": cumulative,
            "Investment": investment,
        }
    )
    return df, payback_month


def scenario_curve_static(
    days_per_month: float,
    ticket: float,
    ticket_infl: float,
    max_ticket_price: float,
    cogs_pct: float,
    labor_pct: float,
    other_pct: float,
    rent_month_y1: float,
    rent_infl: float,
    cogs_infl: float,
    labor_infl: float,
    other_infl: float,
    min_employees: int,
    hourly_rate: float,
    labor_burden_mult: float,
    hours_per_employee_month: float,
    investment: float,
    min_orders: int = 50,
    max_orders: int = 400,
    step: int = 10,
):
    """
    Static (non-ramp) scenario curve:
    For each constant daily order level, compute 3-year-average MONTHLY profit and payback in months.
    Uses:
      • ticket inflation by year (capped at max_ticket_price)
      • percentage-based COGS & other
      • real rent with inflation
      • labor as max(percentage-based, staffing floor) with inflation
    """
    orders_list = list(range(min_orders, max_orders + 1, step))
    avg_profits = []
    paybacks = []

    base_min_labor = min_employees * hourly_rate * labor_burden_mult * hours_per_employee_month

    for ords in orders_list:
        # Year 1
        ticket_y1 = ticket * (1 + ticket_infl) ** 0
        ticket_y1 = min(ticket_y1, max_ticket_price)
        revenue_y1 = ords * days_per_month * ticket_y1

        cogs_y1 = revenue_y1 * cogs_pct
        other_y1 = revenue_y1 * other_pct
        pct_labor_y1 = revenue_y1 * labor_pct
        min_labor_y1 = base_min_labor
        labor_y1 = max(pct_labor_y1, min_labor_y1)
        rent_y1 = rent_month_y1
        profit_y1 = revenue_y1 - (cogs_y1 + labor_y1 + other_y1 + rent_y1)

        # Year 2
        ticket_y2 = ticket * (1 + ticket_infl) ** 1
        ticket_y2 = min(ticket_y2, max_ticket_price)
        revenue_y2 = ords * days_per_month * ticket_y2

        cogs_y2 = revenue_y2 * (cogs_pct * (1 + cogs_infl))
        other_y2 = revenue_y2 * (other_pct * (1 + other_infl))
        pct_labor_y2 = revenue_y2 * (labor_pct * (1 + labor_infl))
        min_labor_y2 = base_min_labor * (1 + labor_infl)
        labor_y2 = max(pct_labor_y2, min_labor_y2)
        rent_y2 = rent_month_y1 * (1 + rent_infl)
        profit_y2 = revenue_y2 - (cogs_y2 + labor_y2 + other_y2 + rent_y2)

        # Year 3
        ticket_y3 = ticket * (1 + ticket_infl) ** 2
        ticket_y3 = min(ticket_y3, max_ticket_price)
        revenue_y3 = ords * days_per_month * ticket_y3

        cogs_y3 = revenue_y3 * (cogs_pct * (1 + cogs_infl) ** 2)
        other_y3 = revenue_y3 * (other_pct * (1 + other_infl) ** 2)
        pct_labor_y3 = revenue_y3 * (labor_pct * (1 + labor_infl) ** 2)
        min_labor_y3 = base_min_labor * (1 + labor_infl) ** 2
        labor_y3 = max(pct_labor_y3, min_labor_y3)
        rent_y3 = rent_month_y1 * (1 + rent_infl) ** 2
        profit_y3 = revenue_y3 - (cogs_y3 + labor_y3 + other_y3 + rent_y3)

        avg_profit = float(np.mean([profit_y1, profit_y2, profit_y3]))
        avg_profits.append(avg_profit)

        if avg_profit > 0:
            payback = investment / avg_profit
        else:
            payback = float("nan")
        paybacks.append(payback)

    df = pd.DataFrame(
        {
            "Daily Orders (Static)": orders_list,
            "Avg Monthly Profit (3yr, Static)": avg_profits,
            "Payback (Months, Static)": paybacks,
        }
    )
    return df


def pct_status_emoji(actual: float, target: float):
    """
    Color/status helper for percentage metrics.
    actual, target are decimals (0.34 = 34%).
    ±3 percentage points spread for color coding.
    """
    if actual is None or np.isnan(actual) or target is None or np.isnan(target):
        return "⚪"
    # difference in decimal terms (0.03 = 3 percentage points)
    diff = actual - target
    tolerance = 0.03  # ±3 percentage points
    if actual < target - tolerance:
        return "🟢"  # below target range (good - spending less)
    elif target - tolerance <= actual <= target + tolerance:
        return "🟣"  # within target range (on target - purple)
    else:
        return "🔴"  # above target range (bad - spending more)


def profit_status_emoji(net_per_investor: float):
    """
    Color/status helper for net profit per investor (monthly).
    > 0 => green, 0 to -1500 => yellow, < -1500 => red.
    """
    if net_per_investor is None or np.isnan(net_per_investor):
        return "⚪"
    if net_per_investor > 0:
        return "🟢"
    elif net_per_investor >= -1500:
        return "🟡"
    else:
        return "🔴"


# ----------------- Streamlit App (v9) -----------------


def main():
    st.title("Quick Service Restaurant Financial Model Calculator")

    st.write(
        "All business metrics shown are **per month** or **per year** unless otherwise stated.\n\n"
        "*Are you accessing this from a mobile browser? Use the menu button in the upper left to access the adjustment toolbar.*\n"
        "- Revenue, COGS, labor, other OpEx, rent, profit and investor splits are **monthly**.\n"
        "- Inflation inputs are **annual % increases** on ticket price, cost percentages, rent, and misc expenses.\n"
        "- Daily orders ramp from a starting value to a target over a chosen number of months, "
        "then continue to grow at a specified rate until a maximum daily order cap.\n"
        "- Rent is modeled as a real monthly dollar amount with annual increases; occupancy % is Rent ÷ Revenue.\n"
        "- **Misc monthly expenses** can be configured as either a percentage of revenue or a fixed dollar amount, "
        "with annual inflation applied.\n"
        "- **Recommended headcount calculator** helps determine minimum staffing needs based on store operating hours "
        "and employees per shift (assumes 34 hours per employee per week).\n"
        "- Labor is modeled as the **maximum of** a percentage-of-revenue target and a **minimum staffing floor** "
        "(actual employee headcount × hourly rate × burden multiplier × hours per employee per week × 4.33).\n"
        "- Ticket price inflates annually but is capped by a configurable **Max Ticket Price**.\n"
        "- Targets for COGS, Labor, Occupancy (Rent) and Other OpEx are adjustable and used for color coding. "
        "Net Profit is color-coded: green for positive, red for negative."
    )

    # ----- Sidebar: Core Inputs -----
    st.sidebar.header("Core inputs")

    # Ramp growth settings
    st.sidebar.subheader("Order growth – ramp + post-ramp")
    start_orders = st.sidebar.number_input(
        "Start daily orders (month 1)", min_value=0.0, max_value=500.0, value=75.0, step=5.0
    )
    target_orders = st.sidebar.number_input(
        "Target daily orders at ramp end", min_value=1.0, max_value=800.0, value=185.0, step=5.0
    )
    ramp_months = st.sidebar.number_input(
        "Ramp duration (months)", min_value=1, max_value=120, value=24, step=1
    )
    post_ramp_growth_per_month = st.sidebar.slider(
        "Post-ramp growth (additional daily orders per month)", min_value=0.0, max_value=20.0, value=10.0, step=0.5
    )
    max_daily_orders = st.sidebar.number_input(
        "Max daily orders cap", min_value=50.0, max_value=1000.0, value=300.0, step=10.0
    )

    days_per_month = st.sidebar.slider(
        "Operating days per month", min_value=20, max_value=31, value=30
    )
    ticket = st.sidebar.slider(
        "Base ticket ($ per order, year 1)", min_value=8.0, max_value=40.0, value=17.0, step=0.5
    )
    ticket_infl = st.sidebar.slider(
        "Ticket price inflation (annual %)", min_value=0.0, max_value=0.10, value=0.02, step=0.005
    )
    max_ticket_price = st.sidebar.number_input(
        "Max ticket price ($)", min_value=10.0, max_value=100.0, value=34.0, step=1.0
    )

    st.sidebar.subheader("Cost structure – year 1 (% of monthly revenue)")
    cogs_pct = st.sidebar.slider("COGS % of revenue", 0.20, 0.60, 0.348, 0.01)
    labor_pct = st.sidebar.slider("Labor % target of revenue", 0.10, 0.50, 0.254, 0.01)
    other_pct = st.sidebar.slider("Other OpEx % of revenue", 0.05, 0.30, 0.14, 0.01)
    
    st.sidebar.subheader("Misc monthly expenses")
    misc_expense_type = st.sidebar.radio(
        "Misc expense type",
        ["Percentage of Revenue", "Fixed Dollar Amount"],
        index=0,
        help="Choose whether misc expenses are calculated as a percentage of revenue or a fixed monthly amount."
    )
    
    if misc_expense_type == "Percentage of Revenue":
        misc_expense_pct = st.sidebar.slider(
            "Misc expenses % of revenue", 
            min_value=0.0, 
            max_value=0.20, 
            value=0.0, 
            step=0.005,
            help="Misc expenses as a percentage of monthly revenue"
        )
        misc_expense_fixed = 0.0
    else:
        misc_expense_fixed = st.sidebar.number_input(
            "Misc expenses fixed amount ($/month)",
            min_value=0.0,
            max_value=50000.0,
            value=0.0,
            step=100.0,
            help="Fixed monthly misc expenses in dollars"
        )
        misc_expense_pct = 0.0
    
    misc_expense_infl = st.sidebar.slider(
        "Misc expenses inflation (annual %)", 
        min_value=0.0, 
        max_value=0.10, 
        value=0.01, 
        step=0.005,
        help="Annual inflation rate for misc expenses"
    )

    st.sidebar.subheader("Rent (occupancy) – real dollars")
    rent_month_y1 = st.sidebar.number_input(
        "Monthly rent – year 1 ($)", min_value=2000.0, max_value=30000.0, value=10500.0, step=500.0
    )
    rent_infl = st.sidebar.slider(
        "Rent inflation (annual %)", min_value=0.0, max_value=0.10, value=0.03, step=0.005
    )

    st.sidebar.subheader("Recommended headcount calculator")
    st.sidebar.write("Enter store hours for each day of the week:")
    store_hours_mon = st.sidebar.number_input("Monday hours", min_value=0.0, max_value=24.0, value=14.0, step=0.5)
    store_hours_tue = st.sidebar.number_input("Tuesday hours", min_value=0.0, max_value=24.0, value=14.0, step=0.5)
    store_hours_wed = st.sidebar.number_input("Wednesday hours", min_value=0.0, max_value=24.0, value=14.0, step=0.5)
    store_hours_thu = st.sidebar.number_input("Thursday hours", min_value=0.0, max_value=24.0, value=14.0, step=0.5)
    store_hours_fri = st.sidebar.number_input("Friday hours", min_value=0.0, max_value=24.0, value=17.0, step=0.5)
    store_hours_sat = st.sidebar.number_input("Saturday hours", min_value=0.0, max_value=24.0, value=17.0, step=0.5)
    store_hours_sun = st.sidebar.number_input("Sunday hours", min_value=0.0, max_value=24.0, value=14.0, step=0.5)
    
    # Calculate total store hours per week
    total_store_hours_per_week = store_hours_mon + store_hours_tue + store_hours_wed + store_hours_thu + store_hours_fri + store_hours_sat + store_hours_sun
    
    # Calculate store hours per month
    total_store_hours_per_month = total_store_hours_per_week * 4.33
    
    # Display calculated store hours per week and per month
    st.sidebar.info(
        f"**Calculated hours store will be open:**\n"
        f"- Per week: {total_store_hours_per_week:,.1f} hours\n"
        f"- Per month: {total_store_hours_per_month:,.1f} hours"
    )
    
    # Employees per shift input
    employees_per_shift = st.sidebar.number_input(
        "Employees per shift",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
        help="Number of employees needed at all times during store operating hours"
    )
    
    # Calculate minimum employees needed
    # Total employee-hours needed per week = store hours × employees per shift
    # Minimum employees = (total employee-hours needed) / (max hours per employee per week)
    total_employee_hours_needed_per_week = total_store_hours_per_week * employees_per_shift
    max_hours_per_employee = 34.0  # 34 hours per week
    min_employees_calculated = np.ceil(total_employee_hours_needed_per_week / max_hours_per_employee) if max_hours_per_employee > 0 else 0
    min_employees_calculated = int(min_employees_calculated) if min_employees_calculated > 0 else 1
    
    # Calculate total hours to pay out
    total_employee_hours_needed_per_month = total_employee_hours_needed_per_week * 4.33
    
    # Display calculated minimum
    st.sidebar.info(
        f"***Calculated recommended headcount:** {min_employees_calculated} employees\n\n"
        f"****Total staffing-hours needed per month:**\n"
        f"- Per week: {total_employee_hours_needed_per_week:,.1f} hours\n"
        f"- Per month: {total_employee_hours_needed_per_month:,.1f} hours"
    )
    
    st.sidebar.subheader("Labor – Actual employee headcount")
    
    # Initialize session state for actual_employees if not exists
    if 'actual_employees' not in st.session_state:
        st.session_state.actual_employees = min_employees_calculated
    
    # Initialize session state for hours_per_employee_week if not exists
    if 'hours_per_employee_week' not in st.session_state:
        st.session_state.hours_per_employee_week = 34.0
    
    # Button to sync actual headcount to calculated minimum
    if st.sidebar.button("Set to calculated minimum", use_container_width=True):
        st.session_state.actual_employees = min_employees_calculated
        st.session_state.hours_per_employee_week = 34.0  # Match the calculated minimum assumption
    
    actual_employees = st.sidebar.number_input(
        "Actual employee headcount",
        min_value=1,
        max_value=50, 
        value=st.session_state.actual_employees, 
        step=1,
        key="actual_employees",
        help=f"One employee per shift = +/- headcount by {employees_per_shift}"
    )
    
    hourly_rate = st.sidebar.slider(
        "Base hourly wage ($/hr)", min_value=10.0, max_value=40.0, value=18.0, step=0.5
    )
    labor_burden_mult = st.sidebar.slider(
        "Labor burden multiplier (taxes, benefits, etc.)", min_value=1.0, max_value=2.0, value=1.249, step=0.01
    )
    hours_per_employee_week = st.sidebar.number_input(
        "Hours per employee per week", 
        min_value=10.0, 
        value=st.session_state.hours_per_employee_week, 
        step=0.5,
        key="hours_per_employee_week"
    )
    # Convert to monthly using 4.33 weeks/month
    hours_per_employee_month = hours_per_employee_week * 4.33
    
    # Use actual_employees for the labor floor calculation (keeping variable name for compatibility)
    min_employees = actual_employees

    st.sidebar.subheader("Annual inflation on cost percentages")
    cogs_infl = st.sidebar.slider("COGS inflation (annual, % of COGS%)", 0.0, 0.10, 0.015, 0.005)
    labor_infl = st.sidebar.slider("Labor inflation (annual, % of labor%)", 0.0, 0.10, 0.02, 0.005)
    other_infl = st.sidebar.slider("Other OpEx inflation (annual, % of other%)", 0.0, 0.10, 0.01, 0.005)

    st.sidebar.subheader("Investment & investor structure")
    investment = st.sidebar.number_input(
        "Total investment ($)", min_value=50000.0, max_value=2000000.0, value=250000.0, step=25000.0
    )
    num_investors = st.sidebar.number_input(
        "Number of investors (equal split)", min_value=1, max_value=10, value=3, step=1
    )

    st.sidebar.subheader("Performance targets (for color coding)")
    st.sidebar.markdown(
        "**Color key:**<br>"
        "🟣 Purple = On target (±3%)<br>"
        "🟢 Green = Below target (good)<br>"
        "🔴 Red = Above target (bad)",
        unsafe_allow_html=True
    )
    cogs_target = st.sidebar.slider("COGS target percent of revenue", 0.10, 0.60, 0.34, 0.01)
    labor_target = st.sidebar.slider("Labor target percent of revenue", 0.10, 0.50, 0.24, 0.01)
    occ_target = st.sidebar.slider("Rent/occupancy target percent of revenue", 0.05, 0.25, 0.10, 0.01)
    other_target = st.sidebar.slider("Other OpEx target percent of revenue", 0.05, 0.30, 0.15, 0.01)

    # ----- Core 10-year projection using ramp + real rent + real labor -----
    df_5yr, payback_month_5yr = compute_5yr_projection(
        start_orders,
        target_orders,
        int(ramp_months),
        days_per_month,
        ticket,
        ticket_infl,
        max_ticket_price,
        cogs_pct,
        labor_pct,
        other_pct,
        rent_month_y1,
        rent_infl,
        cogs_infl,
        labor_infl,
        other_infl,
        min_employees,
        hourly_rate,
        labor_burden_mult,
        hours_per_employee_month,
        investment,
        post_ramp_growth_per_month,
        max_daily_orders,
        misc_expense_pct=misc_expense_pct,
        misc_expense_fixed=misc_expense_fixed,
        misc_expense_infl=misc_expense_infl,
        total_months=120,
    )

    # Yearly averages (monthly metrics) including dollar costs
    df_year_summary = (
        df_5yr.groupby("Year")
        .agg(
            {
                "Daily Orders": "mean",
                "Monthly Revenue": "mean",
                "COGS (Monthly)": "mean",
                "Labor (Monthly)": "mean",
                "Labor (Target)": "mean",
                "Labor (Actual)": "mean",
                "Other OpEx (Monthly)": "mean",
                "Rent (Monthly)": "mean",
                "Monthly Profit": "mean",
                "Monthly Gross Profit": "mean",
                "Occupancy % (Rent/Revenue)": "mean",
                "Labor % (Actual)": "mean",
            }
        )
        .rename(
            columns={
                "Daily Orders": "Avg Daily Orders",
                "Monthly Revenue": "Avg Monthly Revenue",
                "COGS (Monthly)": "Avg COGS (Monthly)",
                "Labor (Monthly)": "Avg Labor (Monthly)",
                "Labor (Target)": "Avg Labor (Target)",
                "Labor (Actual)": "Avg Labor (Actual)",
                "Other OpEx (Monthly)": "Avg Other OpEx (Monthly)",
                "Rent (Monthly)": "Avg Rent (Monthly)",
                "Monthly Profit": "Avg Monthly Net Profit",
                "Monthly Gross Profit": "Avg Monthly Gross Profit",
                "Occupancy % (Rent/Revenue)": "Avg Occupancy % (Rent/Revenue)",
                "Labor % (Actual)": "Avg Labor % (Actual)",
            }
        )
        .reset_index()
    )

    # Build investor-by-year summary
    df_inv_year = df_year_summary.copy()
    df_inv_year["Gross Profit per Investor (Avg Monthly)"] = (
        df_inv_year["Avg Monthly Gross Profit"] / num_investors
    )
    df_inv_year["Net Profit per Investor (Avg Monthly)"] = (
        df_inv_year["Avg Monthly Net Profit"] / num_investors
    )

    # Build investor-by-month detailed table
    df_inv_month = df_5yr.copy()
    df_inv_month["Gross Profit per Investor (Monthly)"] = df_inv_month["Monthly Gross Profit"] / num_investors
    df_inv_month["Net Profit per Investor (Monthly)"] = df_inv_month["Monthly Profit"] / num_investors

    # Specific months for quick view
    row_m1 = df_5yr[df_5yr["Month"] == 1].iloc[0]
    ramp_end_month = int(ramp_months)
    if (df_5yr["Month"] == ramp_end_month).any():
        row_ramp_end = df_5yr[df_5yr["Month"] == ramp_end_month].iloc[0]
    else:
        row_ramp_end = df_5yr.iloc[-1]
    
    # Get Month 6, 12, and 24 data
    if (df_5yr["Month"] == 6).any():
        row_m6 = df_5yr[df_5yr["Month"] == 6].iloc[0]
    else:
        row_m6 = df_5yr.iloc[-1]
    
    if (df_5yr["Month"] == 12).any():
        row_m12 = df_5yr[df_5yr["Month"] == 12].iloc[0]
    else:
        row_m12 = df_5yr.iloc[-1]
    
    if (df_5yr["Month"] == 24).any():
        row_m24 = df_5yr[df_5yr["Month"] == 24].iloc[0]
    else:
        row_m24 = df_5yr.iloc[-1]

    # Year 1 averages (for dashboard top investor view)
    year1 = df_inv_year[df_inv_year["Year"] == 1]
    if not year1.empty:
        y1_avg_gross = float(year1["Avg Monthly Gross Profit"].iloc[0])
        y1_avg_net = float(year1["Avg Monthly Net Profit"].iloc[0])
        y1_avg_occ = float(year1["Avg Occupancy % (Rent/Revenue)"].iloc[0])
        y1_avg_labor_pct = float(year1["Avg Labor % (Actual)"].iloc[0])
    else:
        y1_avg_gross = float(df_5yr["Monthly Gross Profit"].mean())
        y1_avg_net = float(df_5yr["Monthly Profit"].mean())
        y1_avg_occ = float(df_5yr["Occupancy % (Rent/Revenue)"].mean())
        y1_avg_labor_pct = float(df_5yr["Labor % (Actual)"].mean())

    gross_per_investor_y1 = y1_avg_gross / num_investors if num_investors > 0 else float("nan")
    net_per_investor_y1 = y1_avg_net / num_investors if num_investors > 0 else float("nan")

    # ----- Top-level metrics -----
    st.subheader("Key monthly metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Month 1 Revenue ($ / month)",
        f"${row_m1['Monthly Revenue']:,.0f}",
        help="Monthly revenue in the very first month with low daily orders.",
    )
    col2.metric(
        f"Month {ramp_end_month} Revenue ($ / month)",
        f"${row_ramp_end['Monthly Revenue']:,.0f}",
        help="Monthly revenue at the end of the ramp period (target daily orders plus any ticket inflation).",
    )
    if payback_month_5yr is not None:
        col3.metric(
            "Payback (Months, ramp + real rent + real labor)",
            f"{payback_month_5yr:.1f}",
            help="First month where cumulative profit exceeds total investment.",
        )
    else:
        col3.metric("Payback (Months, ramp + real rent + real labor)", "N/A")

    # Occupancy & Revenue snapshot
    st.markdown("---")
    st.subheader("Occupancy & revenue snapshot")

    # Yearly average net revenue row (Years 1-4)
    col_y1, col_y2, col_y3, col_y4 = st.columns(4)
    
    for year_num, col in zip([1, 2, 3, 4], [col_y1, col_y2, col_y3, col_y4]):
        year_data = df_year_summary[df_year_summary["Year"] == year_num]
        if not year_data.empty:
            avg_net_revenue = float(year_data["Avg Monthly Net Profit"].iloc[0])
            col.metric(
                f"Year {year_num} avg net revenue ($ / month)",
                f"${avg_net_revenue:,.0f}",
                help=f"Average monthly net profit (Revenue - all costs) for Year {year_num}.",
            )
        else:
            col.metric(
                f"Year {year_num} avg net revenue ($ / month)",
                "N/A",
            )
    
    st.markdown("---")  # Separator before monthly breakdown

    # Calculate monthly orders for all months
    monthly_orders_m1 = float(row_m1["Daily Orders"]) * days_per_month
    monthly_orders_m6 = float(row_m6["Daily Orders"]) * days_per_month
    monthly_orders_m12 = float(row_m12["Daily Orders"]) * days_per_month
    monthly_orders_m24 = float(row_m24["Daily Orders"]) * days_per_month
    
    # Month 1 metrics
    col_m1_1, col_m1_2, col_m1_3, col_m1_4 = st.columns(4)
    col_m1_1.metric(
        "Month 1 occupancy ($ / month)",
        f"${row_m1['Rent (Monthly)']:,.0f}",
        help="Monthly rent (occupancy cost) in month 1.",
    )
    col_m1_2.metric(
        "Month 1 revenue ($ / month)",
        f"${row_m1['Monthly Revenue']:,.0f}",
        help="Monthly revenue in Month 1.",
    )
    col_m1_3.metric(
        "Month 1 Monthly Orders",
        f"{monthly_orders_m1:,.0f}",
        help="Total number of orders in Month 1 (Daily Orders × Operating Days).",
    )
    col_m1_4.metric(
        "Month 1 Net Revenue ($ / month)",
        f"${row_m1['Monthly Profit']:,.0f}",
        help="Monthly net profit (Revenue - all costs) in Month 1.",
    )
    
    # Month 6 metrics
    col_m6_1, col_m6_2, col_m6_3, col_m6_4 = st.columns(4)
    col_m6_1.metric(
        "Month 6 Occupancy ($ / month)",
        f"${row_m6['Rent (Monthly)']:,.0f}",
        help="Monthly rent (occupancy cost) in Month 6.",
    )
    col_m6_2.metric(
        "Month 6 Revenue ($ / month)",
        f"${row_m6['Monthly Revenue']:,.0f}",
        help="Monthly revenue in Month 6.",
    )
    col_m6_3.metric(
        "Month 6 Monthly Orders",
        f"{monthly_orders_m6:,.0f}",
        help="Total number of orders in Month 6 (Daily Orders × Operating Days).",
    )
    col_m6_4.metric(
        "Month 6 Net Revenue ($ / month)",
        f"${row_m6['Monthly Profit']:,.0f}",
        help="Monthly net profit (Revenue - all costs) in Month 6.",
    )
    
    # Month 12 metrics
    col_m12_1, col_m12_2, col_m12_3, col_m12_4 = st.columns(4)
    col_m12_1.metric(
        "Month 12 Occupancy ($ / month)",
        f"${row_m12['Rent (Monthly)']:,.0f}",
        help="Monthly rent (occupancy cost) in Month 12.",
    )
    col_m12_2.metric(
        "Month 12 Revenue ($ / month)",
        f"${row_m12['Monthly Revenue']:,.0f}",
        help="Monthly revenue in Month 12.",
    )
    col_m12_3.metric(
        "Month 12 Monthly Orders",
        f"{monthly_orders_m12:,.0f}",
        help="Total number of orders in Month 12 (Daily Orders × Operating Days).",
    )
    col_m12_4.metric(
        "Month 12 Net Revenue ($ / month)",
        f"${row_m12['Monthly Profit']:,.0f}",
        help="Monthly net profit (Revenue - all costs) in Month 12.",
    )
    
    # Month 24 metrics
    col_m24_1, col_m24_2, col_m24_3, col_m24_4 = st.columns(4)
    col_m24_1.metric(
        "Month 24 Occupancy ($ / month)",
        f"${row_m24['Rent (Monthly)']:,.0f}",
        help="Monthly rent (occupancy cost) in Month 24.",
    )
    col_m24_2.metric(
        "Month 24 Revenue ($ / month)",
        f"${row_m24['Monthly Revenue']:,.0f}",
        help="Monthly revenue in Month 24.",
    )
    col_m24_3.metric(
        "Month 24 Monthly Orders",
        f"{monthly_orders_m24:,.0f}",
        help="Total number of orders in Month 24 (Daily Orders × Operating Days).",
    )
    col_m24_4.metric(
        "Month 24 Net Revenue ($ / month)",
        f"${row_m24['Monthly Profit']:,.0f}",
        help="Monthly net profit (Revenue - all costs) in Month 24.",
    )
    
    # ----- Labor Headcount & Comparison -----
    st.markdown("---")
    st.subheader("Labor Headcount & Comparison")
    
    # Calculate Year 1 average labor budget (target) for affordability calculation
    # Use Labor (Target) - the percentage-based budget, not the actual spend
    year1_labor_budget = float(df_5yr[df_5yr["Year"] == 1]["Labor (Target)"].mean()) if not df_5yr[df_5yr["Year"] == 1].empty else 0
    
    # Total hours needed to staff the store (store hours/week × employees per shift)
    total_hours_needed_per_week = total_store_hours_per_week * employees_per_shift
    total_hours_needed_per_month = total_hours_needed_per_week * 4.33
    
    # Hours we can afford based on Year 1 average labor budget (target)
    # Labor cost = hours × hourly_rate × burden_multiplier
    # So: hours = labor_cost / (hourly_rate × burden_multiplier)
    hours_affordable_per_month = year1_labor_budget / (hourly_rate * labor_burden_mult) if (hourly_rate * labor_burden_mult) > 0 else 0
    hours_affordable_per_week = hours_affordable_per_month / 4.33
    
    # Calculate employee costs
    cost_per_hour = hourly_rate * labor_burden_mult
    hours_per_employee_month = hours_per_employee_week * 4.33
    # Total baseline employee cost needed per month = Actual employee headcount × hours per employee per month × cost per hour
    total_baseline_employee_cost_per_month = actual_employees * hours_per_employee_month * cost_per_hour
    # Expected employee cost per month = actual employees × hours per employee per month × cost per hour
    expected_employee_cost_per_month = actual_employees * hours_per_employee_month * cost_per_hour
    
    col_comp1, col_comp2 = st.columns(2)
    
    # Comparison 1: Actual vs Minimum Headcount
    headcount_diff = actual_employees - min_employees_calculated
    if headcount_diff >= 0:
        headcount_status = f"✅ {headcount_diff} above minimum"
        headcount_color = "normal"
    else:
        headcount_status = f"⚠️ {abs(headcount_diff)} below minimum"
        headcount_color = "inverse"
    
    col_comp1.metric(
        "Headcount Comparison",
        f"{actual_employees} actual vs {min_employees_calculated} recommended",
        headcount_status,
        help=f"Actual employee headcount ({actual_employees}) compared to calculated minimum ({min_employees_calculated}) based on store hours."
    )
    
    # Comparison 2: Total Hours Needed
    col_comp2.metric(
        "Total Hours Needed (per week)",
        f"{total_hours_needed_per_week:,.1f} hrs",
        help=f"Total employee-hours needed per week to staff store with {employees_per_shift} employees at all times ({total_store_hours_per_week:.1f} store hours/week × {employees_per_shift} employees)."
    )
    
    # Detailed breakdown
    st.markdown("#### Detailed Breakdown")
    comparison_df = pd.DataFrame({
        "Metric": [
            "Actual employee headcount",
            "Calculated recommended headcount",
            "Total store hours per week",
            "Total staffing-hours needed per week",
            "Total staffing-hours needed per Month",
            "Total calculated employee cost needed per month per recommended headcount",
            "Total actual employee cost per month per actual inputed headcount",
        ],
        "Value": [
            f"{actual_employees} employees",
            f"{min_employees_calculated} employees",
            f"{total_store_hours_per_week:.1f} hours",
            f"{total_hours_needed_per_week:.1f} hours",
            f"{total_hours_needed_per_month:.1f} hours",
            f"${total_baseline_employee_cost_per_month:,.2f}",
            f"${expected_employee_cost_per_month:,.2f}",
        ]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Table showing months 1-120
    st.markdown("### Monthly Breakdown (Months 1-120)")
    months_1_120 = df_5yr[df_5yr["Month"] <= 120].copy()
    months_1_120["Monthly Orders"] = months_1_120["Daily Orders"] * days_per_month
    months_1_120["Weekly Orders"] = months_1_120["Daily Orders"] * 7
    monthly_breakdown_df = months_1_120[["Month", "Rent (Monthly)", "Monthly Revenue", "Monthly Profit", "Daily Orders", "Weekly Orders", "Monthly Orders"]].copy()
    monthly_breakdown_df.columns = ["Month", "Occupancy ($)", "Revenue ($)", "Net Revenue ($)", "Daily Orders", "Weekly Orders", "Monthly Orders"]
    st.dataframe(
        monthly_breakdown_df.style.format(
            {
                "Occupancy ($)": "${:,.0f}",
                "Revenue ($)": "${:,.0f}",
                "Net Revenue ($)": "${:,.0f}",
                "Daily Orders": "{:,.1f}",
                "Weekly Orders": "{:,.1f}",
                "Monthly Orders": "{:,.0f}",
            }
        ),
        use_container_width=True,
        height=400
    )

    # Yearly cost breakdown table (Years 1-10)
    st.markdown("### Yearly Budgeted Cost Breakdown ($ / month)")
    
    # Build dictionary with data for each year
    comp_data = {
        "Metric": [
            "Revenue",
            "COGS",
            "Labor (Budgeted)",
            "Labor (Actual)",
            "Other OpEx",
            "Rent (Budgeted)",
            "Rent (Actual)",
            "Gross Profit",
            "Net Profit",
        ]
    }
    
    # Format dictionary for styling
    format_dict = {}
    
    # Add columns for Years 1-10
    for year in range(1, 11):
        year_data = df_year_summary[df_year_summary["Year"] == year]
        if not year_data.empty:
            avg_revenue = float(year_data["Avg Monthly Revenue"].iloc[0])
            rent_budgeted = avg_revenue * occ_target  # Target rent based on occupancy target %
            comp_data[f"Year {year}"] = [
                avg_revenue,
                float(year_data["Avg COGS (Monthly)"].iloc[0]),
                float(year_data["Avg Labor (Target)"].iloc[0]),
                float(year_data["Avg Labor (Actual)"].iloc[0]),
                float(year_data["Avg Other OpEx (Monthly)"].iloc[0]),
                rent_budgeted,
                float(year_data["Avg Rent (Monthly)"].iloc[0]),
                float(year_data["Avg Monthly Gross Profit"].iloc[0]),
                float(year_data["Avg Monthly Net Profit"].iloc[0]),
            ]
            format_dict[f"Year {year}"] = "{:,.0f}"
        else:
            # If year doesn't exist, fill with zeros or NaN
            comp_data[f"Year {year}"] = [0.0] * 9
            format_dict[f"Year {year}"] = "{:,.0f}"
    
    comp_df = pd.DataFrame(comp_data)
    
    # Function to apply color coding to Labor, Rent, and Net Profit rows
    def color_labor_and_rent_rows(row):
        """Apply color coding to Labor (Target), Labor (Actual), Rent (Actual), and Net Profit rows"""
        colors = [''] * len(row)
        
        # Find row indices
        # 0: Revenue, 1: COGS, 2: Labor (Budgeted), 3: Labor (Actual), 4: Other OpEx, 
        # 5: Rent (Budgeted), 6: Rent (Actual), 7: Gross Profit, 8: Net Profit
        
        if row.name == 2:  # Labor (Budgeted) row
            return [''] * len(row)  # No color for budgeted row
        elif row.name == 3:  # Labor (Actual) row
            # Compare Actual vs Budgeted for each year column
            for i, col in enumerate(comp_df.columns):
                if col.startswith('Year '):
                    try:
                        budgeted_idx = 2  # Labor (Budgeted) row index
                        actual_val = row[col]
                        target_val = comp_df.loc[budgeted_idx, col]
                        
                        if target_val > 0:
                            diff_pct = (actual_val - target_val) / target_val
                            tolerance = 0.03  # ±3%
                            
                            if diff_pct < -tolerance:
                                colors[i] = 'color: #228B22'  # Green (good - spending less)
                            elif -tolerance <= diff_pct <= tolerance:
                                colors[i] = 'color: #9370DB'  # Purple (on target)
                            else:
                                colors[i] = 'color: #DC143C'  # Red (bad - spending more)
                    except (ValueError, TypeError, KeyError):
                        pass
            return colors
        elif row.name == 5:  # Rent (Budgeted) row
            return [''] * len(row)  # No color for budgeted row
        elif row.name == 6:  # Rent (Actual) row
            # Compare Actual Rent vs Budgeted Rent for each year column
            for i, col in enumerate(comp_df.columns):
                if col.startswith('Year '):
                    try:
                        budgeted_idx = 5  # Rent (Budgeted) row index
                        actual_rent = row[col]
                        budgeted_rent = comp_df.loc[budgeted_idx, col]
                        
                        if budgeted_rent > 0:
                            diff_pct = (actual_rent - budgeted_rent) / budgeted_rent
                            tolerance = 0.03  # ±3%
                            
                            if diff_pct < -tolerance:
                                colors[i] = 'color: #228B22'  # Green (good - spending less than budgeted)
                            elif -tolerance <= diff_pct <= tolerance:
                                colors[i] = 'color: #9370DB'  # Purple (on target)
                            else:
                                colors[i] = 'color: #DC143C'  # Red (bad - spending more than budgeted)
                    except (ValueError, TypeError, KeyError, ZeroDivisionError):
                        pass
            return colors
        elif row.name == 8:  # Net Profit row
            # Color code Net Profit: green for positive, red for negative
            for i, col in enumerate(comp_df.columns):
                if col.startswith('Year '):
                    try:
                        net_profit_val = row[col]
                        if net_profit_val > 0:
                            colors[i] = 'color: #228B22'  # Green (positive profit)
                        elif net_profit_val < 0:
                            colors[i] = 'color: #DC143C'  # Red (negative profit)
                        # If exactly 0, leave default (no color)
                    except (ValueError, TypeError, KeyError):
                        pass
            return colors
        else:
            return [''] * len(row)
    
    # Apply styling with color coding
    styled_df = comp_df.style.format(format_dict).apply(color_labor_and_rent_rows, axis=1)
    
    st.dataframe(
        styled_df,
        use_container_width=True
    )
    
    # Add note about Other OpEx including Misc Expenses
    st.caption(
        "💡 **Note:** Other OpEx includes Misc Monthly Expenses (either percentage-based or fixed dollar amount)."
    )

    # Find first profitable year
    first_profitable_year = None
    for year in range(1, 11):
        year_data = df_year_summary[df_year_summary["Year"] == year]
        if not year_data.empty:
            avg_profit = float(year_data["Avg Monthly Net Profit"].iloc[0])
            if avg_profit > 0:
                first_profitable_year = year
                break
    
    # Calculate full staffing cost: Total Labor Hours to Pay Out per month × (hourly_rate × labor_burden_mult)
    cost_per_hour = hourly_rate * labor_burden_mult
    full_staffing_cost_per_month = total_employee_hours_needed_per_month * cost_per_hour
    
    # Calculate full staffing cost plus one more employee per shift
    # Total Labor Hours to Pay Out per month × cost + Calculated Hours Store Will Be Open per month × cost
    full_staffing_plus_one_cost_per_month = total_employee_hours_needed_per_month * cost_per_hour + total_store_hours_per_month * cost_per_hour
    # Calculate extra headcount: extra hours / hours per employee per month
    hours_per_employee_month = hours_per_employee_week * 4.33
    extra_headcount_plus_one = total_store_hours_per_month / hours_per_employee_month if hours_per_employee_month > 0 else 0
    
    # Find first year they can afford full staffing cost
    # Compare against Labor (Target) - the percentage-based budget to see when revenue can cover the cost
    first_year_afford_staffing = None
    for year in range(1, 11):
        year_data = df_year_summary[df_year_summary["Year"] == year]
        if not year_data.empty:
            avg_labor_target = float(year_data["Avg Labor (Target)"].iloc[0])
            if avg_labor_target >= full_staffing_cost_per_month:
                first_year_afford_staffing = year
                break
    
    # Find first year they can afford full staffing plus one more employee (5 employees)
    first_year_afford_staffing_plus_one = None
    for year in range(1, 11):
        year_data = df_year_summary[df_year_summary["Year"] == year]
        if not year_data.empty:
            avg_labor_target = float(year_data["Avg Labor (Target)"].iloc[0])
            if avg_labor_target >= full_staffing_plus_one_cost_per_month:
                first_year_afford_staffing_plus_one = year
                break
    
    # Display new information
    col_info1, col_info2, col_info3 = st.columns(3)
    
    if first_profitable_year is not None:
        col_info1.success(
            f"**First profitable year: Year {first_profitable_year}**\n\n"
            f"Average monthly net profit becomes positive in Year {first_profitable_year}."
        )
    else:
        col_info1.warning(
            "**First profitable year: Not within 10 years :(**\n\n"
            "Average monthly net profit does not become positive within the 10-year projection."
        )
    
    if first_year_afford_staffing is not None:
        col_info2.success(
            f"**First year the business can afford full staffing: Year {first_year_afford_staffing}**\n\n"
            f"Labor budget can cover full staffing cost (${full_staffing_cost_per_month:,.0f}/month) "
            f"starting in Year {first_year_afford_staffing}."
        )
    else:
        col_info2.warning(
            f"**First year the business can afford full staffing: Not within 10 years :(**\n\n"
            f"Labor budget cannot cover full staffing cost (${full_staffing_cost_per_month:,.0f}/month) "
            f"within the 10-year projection."
        )
    
    if first_year_afford_staffing_plus_one is not None:
        col_info3.success(
            f"**First year the business can afford to hire one additional employee per shift: Year {first_year_afford_staffing_plus_one}**\n\n"
            f"Labor budget can cover full staffing plus one more employee per shift "
            f"(${full_staffing_plus_one_cost_per_month:,.0f}/month, equals {extra_headcount_plus_one:.1f} extra employees) starting in Year {first_year_afford_staffing_plus_one}."
        )
    else:
        col_info3.warning(
            f"**First year the business can afford to hire one additional employee per shift: Not within 10 years :(**\n\n"
            f"Labor budget cannot cover full staffing plus one more employee "
            f"(${full_staffing_plus_one_cost_per_month:,.0f}/month) within the 10-year projection."
        )

    # ----- Investor View – Year 1–10 metrics -----
    st.markdown("### Investor View – Average Monthly Profit Splits (Year 1)")

    colg1, colg2, colg3 = st.columns(3)
    colg1.metric(
        "Gross Profit (Avg Monthly, Year 1)",
        f"${y1_avg_gross:,.0f}",
        help="Average **monthly** gross profit (Revenue – COGS) in Year 1.",
    )
    colg2.metric(
        "Gross Profit per Investor (Avg Monthly, Year 1)",
        f"${gross_per_investor_y1:,.0f}",
        help="Average **monthly** gross profit per investor (equal split) in Year 1.",
    )
    colg3.metric(
        "Net Profit per Investor (Avg Monthly, Year 1)",
        f"${net_per_investor_y1:,.0f}",
        help="Average **monthly** net profit (after all costs incl. rent and labor) per investor in Year 1.",
    )

    # Years 1–5 summary blocks with color-coded metrics
    def render_year_summary_block(year_n: int):
        yr_row = df_inv_year[df_inv_year["Year"] == year_n]
        if yr_row.empty:
            return

        avg_revenue = float(yr_row["Avg Monthly Revenue"].iloc[0])
        avg_rent = float(yr_row["Avg Rent (Monthly)"].iloc[0])
        avg_cogs = float(yr_row["Avg COGS (Monthly)"].iloc[0])
        avg_labor = float(yr_row["Avg Labor (Monthly)"].iloc[0])
        avg_other = float(yr_row["Avg Other OpEx (Monthly)"].iloc[0])
        avg_occ_pct = float(yr_row["Avg Occupancy % (Rent/Revenue)"].iloc[0])
        avg_labor_pct = float(yr_row["Avg Labor % (Actual)"].iloc[0])
        avg_net = float(yr_row["Avg Monthly Net Profit"].iloc[0])
        avg_daily_orders = float(yr_row["Avg Daily Orders"].iloc[0])
        net_per_inv = float(yr_row["Net Profit per Investor (Avg Monthly)"].iloc[0])

        # Derived %
        cogs_pct_year = avg_cogs / avg_revenue if avg_revenue > 0 else np.nan
        other_pct_year = avg_other / avg_revenue if avg_revenue > 0 else np.nan

        # Emoji status colors
        occ_emoji = pct_status_emoji(avg_occ_pct, occ_target)
        labor_emoji = pct_status_emoji(avg_labor_pct, labor_target)
        cogs_emoji = pct_status_emoji(cogs_pct_year, cogs_target)
        other_emoji = pct_status_emoji(other_pct_year, other_target)
        profit_emoji = profit_status_emoji(net_per_inv)

        # Wrap summary content in an expander to make it collapsible
        with st.expander(f"Year {year_n} Summary", expanded=False):
            # Rent (always info)
            st.info(f"🏠 Year {year_n} average rent: **${avg_rent:,.0f}/month**.")

            # Occupancy %
            if not np.isnan(avg_occ_pct):
                st.info(
                    f"{occ_emoji} Year {year_n} average occupancy: "
                    f"**{avg_occ_pct*100:,.1f}%** vs target **{occ_target*100:,.1f}%**."
                )
            else:
                st.info(f"{occ_emoji} Year {year_n} average occupancy: **N/A**.")

            # Labor %
            if not np.isnan(avg_labor_pct):
                st.info(
                    f"{labor_emoji} Year {year_n} average labor: "
                    f"**{avg_labor_pct*100:,.1f}%** vs target **{labor_target*100:,.1f}%**."
                )
            else:
                st.info(f"{labor_emoji} Year {year_n} average labor: **N/A**.")

            # COGS %
            if not np.isnan(cogs_pct_year):
                st.info(
                    f"{cogs_emoji} Year {year_n} average COGS: "
                    f"**{cogs_pct_year*100:,.1f}%** vs target **{cogs_target*100:,.1f}%**."
                )
            else:
                st.info(f"{cogs_emoji} Year {year_n} average COGS: **N/A**.")

            # Other OpEx %
            if not np.isnan(other_pct_year):
                st.info(
                    f"{other_emoji} Year {year_n} average Other OpEx: "
                    f"**{other_pct_year*100:,.1f}%** vs target **{other_target*100:,.1f}%**."
                )
            else:
                st.info(f"{other_emoji} Year {year_n} average Other OpEx: **N/A**.")

            # Net per investor
            st.info(
                f"{profit_emoji} Year {year_n} net profit per investor (avg monthly): "
                f"**${net_per_inv:,.0f}**."
            )

            # Story line
            st.info(
                f"📈 In Year {year_n}, each investor averages **${net_per_inv:,.0f}/month net** "
                f"on **{avg_daily_orders:,.1f} daily orders**."
            )

    # Render Year 1–10 summary blocks under investor panels
    render_year_summary_block(1)

    # Years 2–10 investor metrics + summaries
    for year_n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        yr_row = df_inv_year[df_inv_year["Year"] == year_n]
        if yr_row.empty:
            continue

        avg_gross = float(yr_row["Avg Monthly Gross Profit"].iloc[0])
        avg_net = float(yr_row["Avg Monthly Net Profit"].iloc[0])
        gross_per_inv = avg_gross / num_investors if num_investors > 0 else float("nan")
        net_per_inv = avg_net / num_investors if num_investors > 0 else float("nan")

        st.markdown(f"### Investor View – Average Monthly Profit Splits (Year {year_n})")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            f"Gross Profit (Avg Monthly, Year {year_n})",
            f"${avg_gross:,.0f}",
            help=f"Average **monthly** gross profit (Revenue – COGS) in Year {year_n}.",
        )
        c2.metric(
            f"Gross Profit per Investor (Avg Monthly, Year {year_n})",
            f"${gross_per_inv:,.0f}",
            help=f"Average **monthly** gross profit per investor (equal split) in Year {year_n}.",
        )
        c3.metric(
            f"Net Profit per Investor (Avg Monthly, Year {year_n})",
            f"${net_per_inv:,.0f}",
            help=f"Average **monthly** net profit (after all costs incl. rent and labor) per investor in Year {year_n}.",
        )

        # Year summary block directly under this investor panel
        render_year_summary_block(year_n)

    st.caption(
        "All profit and split numbers above are **per month**, not per year. "
        "Rent is modeled as a fixed monthly dollar amount with annual inflation. "
        "Labor is the maximum of a percentage-of-revenue target and a staffing floor (employees × wage × burden × hours/week × 4.33). "
        "Occupancy % is computed as Rent ÷ Revenue; Labor % (Actual) is Labor ÷ Revenue. "
        "Color coding is based on your adjustable targets in the sidebar."
    )

    # ----- Tabs -----
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Ramp & Yearly View",
            "10-Year Projection (Monthly)",
            "Order-Level Sensitivity (Static)",
            "Investor Splits (Yearly & Monthly)",
        ]
    )

    # ---- Tab 1: Ramp & Yearly View ----
    with tab1:
        st.markdown("### Daily Orders – Ramp + Post-Ramp Growth")
        fig_ramp, ax_ramp = plt.subplots()
        ax_ramp.plot(df_5yr["Month"], df_5yr["Daily Orders"])
        ax_ramp.set_xlabel("Month")
        ax_ramp.set_ylabel("Daily Orders")
        ax_ramp.set_title("Daily Orders Over 10 Years (Ramp + Post-Ramp Growth)")
        st.pyplot(fig_ramp)

        st.markdown("### Year-by-Year Average Monthly Metrics (with Dollar Costs)")
        st.dataframe(
            df_year_summary.style.format(
                {
                    "Avg Daily Orders": "{:,.1f}",
                    "Avg Monthly Revenue": "{:,.0f}",
                    "Avg COGS (Monthly)": "{:,.0f}",
                    "Avg Labor (Monthly)": "{:,.0f}",
                    "Avg Other OpEx (Monthly)": "{:,.0f}",
                    "Avg Rent (Monthly)": "{:,.0f}",
                    "Avg Monthly Net Profit": "{:,.0f}",
                    "Avg Monthly Gross Profit": "{:,.0f}",
                    "Avg Occupancy % (Rent/Revenue)": "{:.2%}",
                    "Avg Labor % (Actual)": "{:.2%}",
                }
            )
        )

        st.markdown("#### Year 1 Average Monthly Cost & Profit Breakdown")
        y1_rows = df_5yr[df_5yr["Year"] == 1]
        if not y1_rows.empty:
            avg_cogs_y1 = float(y1_rows["COGS (Monthly)"].mean())
            avg_labor_y1 = float(y1_rows["Labor (Monthly)"].mean())
            avg_other_y1 = float(y1_rows["Other OpEx (Monthly)"].mean())
            avg_rent_y1 = float(y1_rows["Rent (Monthly)"].mean())
            avg_profit_y1 = float(y1_rows["Monthly Profit"].mean())

            fig_y1, ax_y1 = plt.subplots()
            labels = ["COGS", "Labor", "Other OpEx", "Rent", "Net Profit"]
            values = [avg_cogs_y1, avg_labor_y1, avg_other_y1, avg_rent_y1, avg_profit_y1]
            ax_y1.bar(labels, values)
            ax_y1.set_ylabel("Dollars per Month")
            ax_y1.set_title("Year 1 Average Monthly Cost & Profit Breakdown")
            plt.xticks(rotation=30)
            st.pyplot(fig_y1)

    # ---- Tab 2: 10-Year Projection (Monthly) ----
    with tab2:
        st.markdown("### 10-Year Cumulative Profit vs Investment (Monthly Profit)")

        if payback_month_5yr is not None:
            st.write(
                f"Estimated **payback month**: **Month {payback_month_5yr}** "
                f"(~{payback_month_5yr/12:.1f} years)."
            )
        else:
            st.write(
                "Under the current assumptions, cumulative profit does **not** reach the total investment within 10 years."
            )

        fig_cum, ax_cum = plt.subplots()
        ax_cum.plot(df_5yr["Month"], df_5yr["Cumulative Profit"], label="Cumulative Profit")
        ax_cum.plot(df_5yr["Month"], df_5yr["Investment"], label="Total Investment")
        ax_cum.set_xlabel("Month")
        ax_cum.set_ylabel("Dollars")
        ax_cum.set_title("Cumulative Profit vs Investment (10 Years, Monthly Profit)")
        ax_cum.legend()
        st.pyplot(fig_cum)

        st.markdown("#### 10-Year Monthly Table (All Values Are Monthly)")

        st.dataframe(
            df_5yr.style.format(
                {
                    "Daily Orders": "{:,.1f}",
                    "Monthly Revenue": "{:,.0f}",
                    "COGS (Monthly)": "{:,.0f}",
                    "Labor (Monthly)": "{:,.0f}",
                    "Other OpEx (Monthly)": "{:,.0f}",
                    "Rent (Monthly)": "{:,.0f}",
                    "Occupancy % (Rent/Revenue)": "{:.2%}",
                    "Labor % (Actual)": "{:.2%}",
                    "Monthly Profit": "{:,.0f}",
                    "Monthly Gross Profit": "{:,.0f}",
                    "Cumulative Profit": "{:,.0f}",
                    "Investment": "{:,.0f}",
                }
            )
        )

    # ---- Tab 3: Order-Level Sensitivity (Static) ----
    with tab3:
        st.markdown(
            "### Static Order Sensitivity (No Ramp)\n"
            "This ignores the ramp and assumes a **constant** daily order level for 3 years.\n"
            "It still applies ticket inflation (capped), rent inflation, cost inflation, and the labor floor vs % logic."
        )

        df_static = scenario_curve_static(
            days_per_month,
            ticket,
            ticket_infl,
            max_ticket_price,
            cogs_pct,
            labor_pct,
            other_pct,
            rent_month_y1,
            rent_infl,
            cogs_infl,
            labor_infl,
            other_infl,
            min_employees,
            hourly_rate,
            labor_burden_mult,
            hours_per_employee_month,
            investment,
            min_orders=50,
            max_orders=400,
            step=10,
        )

        fig_static, ax_static = plt.subplots()
        ax_static.plot(df_static["Daily Orders (Static)"], df_static["Payback (Months, Static)"])
        ax_static.set_xlabel("Daily Orders (Static, No Ramp)")
        ax_static.set_ylabel("Payback (Months)")
        ax_static.set_title("Payback vs Daily Orders (Static 3-Year Avg Monthly Profit)")
        st.pyplot(fig_static)

        with st.expander("Show static scenario table"):
            st.dataframe(
                df_static.style.format(
                    {
                        "Avg Monthly Profit (3yr, Static)": "{:,.0f}",
                        "Payback (Months, Static)": "{:,.1f}",
                    }
                )
            )

    # ---- Tab 4: Investor Splits (Yearly & Monthly) ----
    with tab4:
        st.markdown("### Average Monthly Investor Splits by Year – Years 1–10")

        st.dataframe(
            df_inv_year.style.format(
                {
                    "Avg Daily Orders": "{:,.1f}",
                    "Avg Monthly Revenue": "{:,.0f}",
                    "Avg COGS (Monthly)": "{:,.0f}",
                    "Avg Labor (Monthly)": "{:,.0f}",
                    "Avg Other OpEx (Monthly)": "{:,.0f}",
                    "Avg Rent (Monthly)": "{:,.0f}",
                    "Avg Monthly Net Profit": "{:,.0f}",
                    "Avg Monthly Gross Profit": "{:,.0f}",
                    "Avg Occupancy % (Rent/Revenue)": "{:.2%}",
                    "Avg Labor % (Actual)": "{:.2%}",
                    "Gross Profit per Investor (Avg Monthly)": "{:,.0f}",
                    "Net Profit per Investor (Avg Monthly)": "{:,.0f}",
                }
            )
        )

        st.markdown("---")
        st.markdown("### Investor Splits by Month (Full 10 Years, 120 Rows)")

        with st.expander("Show all 120 months (detailed per month)"):
            st.dataframe(
                df_inv_month[
                    [
                        "Month",
                        "Year",
                        "Daily Orders",
                        "Monthly Revenue",
                        "Monthly Gross Profit",
                        "Monthly Profit",
                        "Gross Profit per Investor (Monthly)",
                        "Net Profit per Investor (Monthly)",
                        "Occupancy % (Rent/Revenue)",
                        "Labor % (Actual)",
                    ]
                ].style.format(
                    {
                        "Daily Orders": "{:,.1f}",
                        "Monthly Revenue": "{:,.0f}",
                        "Monthly Gross Profit": "{:,.0f}",
                        "Monthly Profit": "{:,.0f}",
                        "Gross Profit per Investor (Monthly)": "{:,.0f}",
                        "Net Profit per Investor (Monthly)": "{:,.0f}",
                        "Occupancy % (Rent/Revenue)": "{:.2%}",
                        "Labor % (Actual)": "{:.2%}",
                    }
                )
            )

        st.markdown("### Monthly Investor Splits by Year")

        for year in sorted(df_inv_month["Year"].unique()):
            with st.expander(f"Year {year} – Monthly Investor Splits"):
                df_year = df_inv_month[df_inv_month["Year"] == year].copy()
                st.dataframe(
                    df_year[
                        [
                            "Month",
                            "Daily Orders",
                            "Monthly Revenue",
                            "Monthly Gross Profit",
                            "Monthly Profit",
                            "Gross Profit per Investor (Monthly)",
                            "Net Profit per Investor (Monthly)",
                            "Occupancy % (Rent/Revenue)",
                            "Labor % (Actual)",
                        ]
                    ].style.format(
                        {
                            "Daily Orders": "{:,.1f}",
                            "Monthly Revenue": "{:,.0f}",
                            "Monthly Gross Profit": "{:,.0f}",
                            "Monthly Profit": "{:,.0f}",
                            "Gross Profit per Investor (Monthly)": "{:,.0f}",
                            "Net Profit per Investor (Monthly)": "{:,.0f}",
                            "Occupancy % (Rent/Revenue)": "{:.2%}",
                            "Labor % (Actual)": "{:.2%}",
                        }
                    )
                )


if __name__ == "__main__":
    main()
