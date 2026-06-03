counted_ids = set()

vehicle_count = 0

def count_vehicle(track_id):

    global vehicle_count

    if track_id not in counted_ids:

        counted_ids.add(track_id)

        vehicle_count += 1

    return vehicle_count

