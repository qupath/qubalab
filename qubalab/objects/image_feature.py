import geojson

class ImageFeature:
    def __init__(self,
                 geometry,
                 classification: Classification = None,
                 name: str = None,
                 measurements: Dict[str, float] = None,
                 object_type: str = types.ANNOTATION,
                 color: Tuple[int, int, int] = None,
                 extra_geometries: Dict[str, Any] = None,
                 id: Union[str, int, uuid.UUID] = None,
                 extra_properties: Dict[str, Any] = None):

        object_id = self._to_id_string(id)
        props = {}
        if classification is not None:
            props['classification'] = classification
        if name is not None:
            props['name'] = name
        if measurements is not None:
            # Can't store NaN properly in JSON, so try to remove
            import math
            props['measurements'] = {k: float(v) for k, v in measurements.items()
                                     if isinstance(k, str) and isinstance(v, (int, float)) and not math.isnan(v)}
        if object_type is not None:
            props['object_type'] = object_type
        if color is not None:
            props['color'] = color
        if extra_geometries is not None:
            props['extra_geometries'] = {k: to_geometry(v) for k, v in extra_geometries.items()}

        if extra_properties is not None:
            props.update(extra_properties)
        super().__init__(geometry=to_geometry(geometry), properties=props, id=object_id)
        self['type'] = 'Feature'
        
    
    @staticmethod
    def create_from_feature(feature: geojson.Feature):
        geometry = ImageFeature._find_property(feature, 'geometry')

        plane = ImageFeature._find_property(feature, 'plane')
        if plane is not None:
            geometry = geometries.to_geometry(geometry, z=getattr(plane, 'z', None), t=getattr(plane, 't', None))

        args = dict(
            geometry=geometry,
            id=ImageFeature._find_property(feature, 'id'),
            classification=ImageFeature._find_property(feature, 'classification'),
            name=ImageFeature._find_property(feature, 'name'),
            color=ImageFeature._find_property(feature, 'color'),
            measurements=ImageFeature._find_property(feature, 'measurements'),
            object_type=ImageFeature._find_property(feature, 'object_type'),
        )

        nucleus_geometry = ImageFeature._find_property(feature, 'nucleusGeometry')
        if nucleus_geometry is not None:
            if plane is not None:
                nucleus_geometry = geometries.to_geometry(nucleus_geometry, z=getattr(plane, 'z', None), t=getattr(plane, 't', None))
            args['extra_geometries'] = dict(nucleus=nucleus_geometry)

        args['extra_properties'] = {k: v for k, v in feature['properties'].items() if k not in args and v is not None}
        return ImageFeature(**args)
    
    @staticmethod
    def _find_property(feature: geojson.Feature, property_name: str, default_value=None):
        if property_name in feature:
            return feature[property_name]
        if 'properties' in feature and property_name in feature['properties']:
            return feature['properties'][property_name]
        return default_value