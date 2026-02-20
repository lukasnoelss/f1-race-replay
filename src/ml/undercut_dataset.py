import pandas as pd
import fastf1

def load_race_session(year, round_number):
    session = fastf1.get_session(year, round_number, 'R')
    session.load(telemetry=False, weather=True) # don't need telemetry or weather
    return session

def extract_pit_stops(session):
    pit_stops = []
    laps = session.laps 
    weather = session.weather_data

    # unique drivers in session: 
    drivers = laps['Driver'].unique()
    
    for driver in drivers: 
        #filter laps for THIS driver
        driver_laps = laps[laps['Driver'] == driver]
        
        # We loop by index so we can peek at the NEXT lap
        for i in range(len(driver_laps)):
            lap = driver_laps.iloc[i]
            
            # If the driver entered the pits on this lap:
            if pd.notna(lap['PitInTime']):
                # 1. Calculate Pit Duration (Peeking at the next lap for OutTime)
                try:
                    next_lap = driver_laps.iloc[i + 1]
                    duration = (next_lap['PitOutTime'] - lap['PitInTime']).total_seconds()
                except:
                    # Case: Retired in pits or last lap
                    duration = 25.0 
                
                # 2. Get Track Temp
                try:
                    temp_time = weather.set_index('Time').index.asof(lap['Time'])
                    current_weather = weather[weather['Time'] == temp_time].iloc[0]
                    t_temp = current_weather['TrackTemp']
                except:
                    t_temp = 25.0 # Safety fallback
                
                stop_data = {
                    'Driver': driver,
                    'LapNumber': lap['LapNumber'],
                    'CompoundBefore': lap['Compound'],
                    'TyreLifeBefore': lap['TyreLife'],
                    'Position': lap['Position'],
                    'PitDuration': round(duration, 3),
                    'TrackTemp': t_temp
                }
                pit_stops.append(stop_data)
    return pd.DataFrame(pit_stops)

def find_undercut_scenarios(pit_stops_df, session, track_name, total_laps):
    scenarios = []
    laps = session.laps 
    compound_map = {
    'SOFT': 1,
    'MEDIUM': 2,
    'HARD': 3,
    'INTERMEDIATE': 4,
    'WET': 5,
    'UNKNOWN': 0
}
    
    #ANTI-BIAS FIX: Calculate the average pit duration for this specific RACE
    # This prevents the model from "knowing" if someone had a slow stop in the future.
    if not pit_stops_df.empty:
        # Filter out massive outliers (stops > 50s usually mean retirement/damage)
        clean_stops = pit_stops_df[pit_stops_df['PitDuration'] < 50]
        session_avg_pit = clean_stops['PitDuration'].mean()
    else:
        session_avg_pit = 24.5 # Global average fallback

    for i, attacker in pit_stops_df.iterrows():
        for j, defender in pit_stops_df.iterrows():
            if i == j:
                continue
            
            lap_diff = defender['LapNumber'] - attacker['LapNumber']
            
            if 1 <= lap_diff <= 3:
                
                if attacker['Position'] > defender['Position']:
                    if attacker['Position'] - defender['Position'] <= 3:
                        result_lap = defender['LapNumber'] + 1
                        res_laps = laps[laps['LapNumber'] == result_lap]

                        try:
                            att_after = res_laps[res_laps['Driver'] == attacker['Driver']]['Position'].iloc[0]
                            def_after = res_laps[res_laps['Driver'] == defender['Driver']]['Position'].iloc[0]
                            outcome = "success" if att_after < def_after else "fail"
                        except:
                            outcome = "unknown"
                        try:    
                            # If defender is in P5, the car in front is P4
                            car_ahead_pos = defender['Position'] - 1
                            
                            if car_ahead_pos >= 1:
                                # Get all positions for this specific lap
                                def_lap_data = laps[laps['LapNumber'] == defender['LapNumber']]
                                # Find who was in that position
                                car_ahead_driver = def_lap_data[def_lap_data['Position'] == car_ahead_pos]['Driver'].iloc[0]

                                def_traffic_gap = calculate_gap(laps, defender['Driver'], car_ahead_driver, defender['LapNumber'])
                            else:
                                # They are leading or in P1? No traffic!
                                def_traffic_gap = 10.0 
                        except:
                            def_traffic_gap = 5.0 # Average traffic

                        gap_before = calculate_gap(laps, attacker['Driver'], defender['Driver'], attacker['LapNumber'] - 1) #on the actual pit lap before the pit stop
                        tire_delta = attacker['TyreLifeBefore'] - defender['TyreLifeBefore']
                        
                        # 🏁 STRATEGY FIX: Pit_Delta is now 0 by default (assuming clean stops)
                        # But we keep Att_Pit_Duration as the session average so the model 
                        # knows the "cost" of a stop at this specific track.
                        pit_delta = 0.0 
                        
                        # DATA CLEANING: Only keep realistic duels (Gap < 3.0 seconds)
                        # We also ignore cases where attacker is already AHEAD (Gap < -1.0)
                        if gap_before > 3.0 or gap_before < -1.0:
                            continue

                        if attacker['PitDuration'] > 100 or defender['PitDuration'] > 100:
                            continue
                        att_hardness = compound_map.get(attacker['CompoundBefore'], 0)
                        def_hardness = compound_map.get(defender['CompoundBefore'], 0)
                        race_progress = attacker['LapNumber'] / total_laps
                        scenario ={
                            'Attacker': attacker['Driver'],
                            'Defender': defender['Driver'],
                            'Track': track_name,
                            'Outcome': outcome,
                            'Pit_Delta': pit_delta,
                            'Gap_Before': gap_before,
                            'Tire_Delta': tire_delta,
                            'Race_Progress': round(race_progress, 3),
                            'Def_Traffic_Gap': def_traffic_gap,
                            'Att_Pit_Duration': round(session_avg_pit, 3),
                            'Def_Pit_Duration': round(session_avg_pit, 3),
                            'Track_Temp': attacker['TrackTemp'],
                            'Att_Hardness': att_hardness,
                            'Def_Hardness': def_hardness,
                            'Stint_Length': attacker['TyreLifeBefore'],
                            'Att_Compound': attacker['CompoundBefore'],
                            'Def_Compound': defender['CompoundBefore'],
                            'Attacker_Lap': attacker['LapNumber'],
                            'Defender_Lap': defender['LapNumber'],
                            'Target_Position': defender['Position'],
                            
                        }
                        scenarios.append(scenario)
    return pd.DataFrame(scenarios)
                        
                    
def calculate_gap(laps, driver1, driver2, lap_num):
    #Find laps for each driver
    d1_lap = laps[(laps['Driver'] == driver1) & (laps['LapNumber'] == lap_num)]
    d2_lap = laps[(laps['Driver'] == driver2) & (laps['LapNumber'] == lap_num)]
    
    if d1_lap.empty or d2_lap.empty:
        return 0.0
    #Get the gap
    # Result is a positive number if driver1 is behind driver2
    gap = (d1_lap['Time'].iloc[0] - d2_lap['Time'].iloc[0]).total_seconds()
    return round(gap,3)
                    
                
            
            
    
if __name__ == "__main__":
    fastf1.Cache.enable_cache('.fastf1-cache')
    
    all_scenarios = []
    
    years = [2020,2021,2022,2023, 2024, 2025] 
    
    for year in years:
        for round_num in range(1, 25): 
            try:
                print(f"--- Processing {year} Round {round_num} ---")
                session = load_race_session(year, round_num)
                track = session.event['EventName']
                total_laps = session.total_laps
                stops = extract_pit_stops(session)
                duels = find_undercut_scenarios(stops, session, track, total_laps)
                
                if not duels.empty:
                    all_scenarios.append(duels)
                    print(f"Found {len(duels)} duels.")
                else:
                    print("ℹNo tactical duels in this race.")
                    
            except Exception as e:
                # This will catch rounds that don't exist or loading errors
                print(f"Skipping {year} Round {round_num}")
                continue
    # 2. Merge everything
    if all_scenarios:
        final_df = pd.concat(all_scenarios)
        final_df.to_csv("undercut_dataset.csv", index=False)
        print(f"\nSUCCESS! Global dataset size: {len(final_df)} samples.")
    else:
        print("\n No data collected!")