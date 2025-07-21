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
