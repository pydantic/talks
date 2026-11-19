"""Tests for the geo risk API. Run with `make test`."""

import os

os.environ['LOGFIRE_SEND_TO_LOGFIRE'] = 'false'

from fastapi.testclient import TestClient

from geo_service.app import app

client = TestClient(app)


def test_list_properties():
    properties = client.get('/properties').json()
    assert len(properties) == 300
    assert properties[0].keys() == {'id', 'address', 'kind', 'units', 'lon', 'lat'}


def test_get_property():
    assert client.get('/properties/P203').json()['address'] == '34 State St'
    assert client.get('/properties/nope').status_code == 404


def test_flood_zone():
    # 34 State St stands in the floodway.
    assert client.get('/properties/P203/flood-zone').json() == {'zone': 'floodway', 'distance_to_1_percent_zone_m': 0}
    # 12 Pearl St is outside every zone, and stays outside the 1% zone when the zones grow by 50 m.
    assert client.get('/properties/P033/flood-zone').json()['zone'] == 'minimal'
    before = client.get('/properties/P033/flood-zone').json()['distance_to_1_percent_zone_m']
    after = client.get('/properties/P033/flood-zone', params={'buffer_m': 50}).json()['distance_to_1_percent_zone_m']
    assert before - after == 50


def test_flood_zone_buffer_moves_properties_into_zones():
    ids = [p['id'] for p in client.get('/properties').json()]

    def in_zone(buffer_m: float) -> int:
        zones = [client.get(f'/properties/{i}/flood-zone', params={'buffer_m': buffer_m}).json()['zone'] for i in ids]
        return sum(zone != 'minimal' for zone in zones)

    assert in_zone(50) > in_zone(0)
    assert client.get('/properties/P001/flood-zone', params={'buffer_m': -1}).status_code == 422


def test_elevation():
    elevation = client.get('/properties/P108/elevation').json()
    assert elevation['height_above_water_m'] == round(elevation['elevation_m'] - elevation['water_elevation_m'], 1)


def test_distance_to_water():
    # 34 State St backs onto the North Branch, where it runs through the town centre.
    assert client.get('/properties/P203/distance-to-water').json() == {
        'distance_m': 0,
        'water_name': 'North Branch Winooski River',
    }


def test_aerial_image():
    response = client.get('/properties/P001/aerial-image')
    assert response.headers['content-type'] == 'image/jpeg'
    assert response.content[:2] == b'\xff\xd8'


def test_map_layers():
    assert client.get('/map/flood-zones').json()['type'] == 'FeatureCollection'
    assert client.get('/map/water').json()['type'] == 'FeatureCollection'
    assert client.get('/map/other').status_code == 422
