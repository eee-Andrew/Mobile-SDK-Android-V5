from jnius import autoclass

# Autoclass DJI MSDK classes
Waypoint = autoclass('dji.sdk.mission.waypoint.Waypoint')
WaypointMissionBuilder = autoclass('dji.sdk.mission.waypoint.WaypointMission$Builder')
MissionControl = autoclass('dji.sdk.mission.MissionControl')


def go_to_waypoint(lat, lon, alt):
    builder = WaypointMissionBuilder().waypointCount(1)
    wp = Waypoint(lat, lon, alt)
    builder.addWaypoint(wp)
    mission = builder.build()

    operator = MissionControl.getInstance().getWaypointMissionOperator()
    error = operator.loadMission(mission)
    if error:
        print('Load mission error:', error)
        return
    error = operator.uploadMission()
    if error:
        print('Upload mission error:', error)
        return
    error = operator.startMission()
    if error:
        print('Start mission error:', error)
    else:
        print('Mission started')


def fly_to_altitude(alt):
    """Fly to the given altitude above the current location."""
    # For testing we use the waypoint mission with a single point at the
    # current coordinates. In a real application you would query the
    # aircraft's current GPS position.
    go_to_waypoint(22.5362, 113.9454, alt)
