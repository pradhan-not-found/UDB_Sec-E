import os
import zipfile
import shutil
import csv
import pandas as pd
from PIL import Image, ExifTags
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from pillow_heif import register_heif_opener
register_heif_opener()

ZIP_PATH = r"d:\New folder (2)\UDB\Assignments 1\UDB.ZIP file.zip"
EXTRACT_DIR = r"d:\New folder (2)\UDB\Assignments 1\temp_images"
OUTPUT_DIR = r"d:\New folder (2)\UDB\Assignments 1\classified_images"
REPORT_CSV = r"d:\New folder (2)\UDB\Assignments 1\metadata_report.csv"

# Adamas University approx bounding box
ADAMAS_MIN_LAT, ADAMAS_MAX_LAT = 22.735, 22.742
ADAMAS_MIN_LON, ADAMAS_MAX_LON = 88.452, 88.460

geolocator = Nominatim(user_agent="udb_assignment1_classifier")

def get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image.getexif()

def get_geotagging(exif):
    if not exif:
        return None

    geotagging = {}
    for (idx, tag) in ExifTags.TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                return None
            for (key, val) in ExifTags.GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]

    return geotagging

def get_decimal_from_dms(dms, ref):
    if not dms or len(dms) < 3:
        return None
        
    try:
        degrees = dms[0]
        minutes = dms[1]
        seconds = dms[2]
        
        # Depending on Pillow version, these might be tuples or IFDRational
        if hasattr(degrees, 'numerator'):
            if degrees.denominator == 0 and minutes.denominator == 0 and seconds.denominator == 0:
                return None
            deg = degrees.numerator / degrees.denominator if degrees.denominator != 0 else 0
            min = minutes.numerator / minutes.denominator if minutes.denominator != 0 else 0
            sec = seconds.numerator / seconds.denominator if seconds.denominator != 0 else 0
        elif isinstance(degrees, tuple):
            if degrees[1] == 0 and minutes[1] == 0 and seconds[1] == 0:
                return None
            deg = degrees[0] / degrees[1] if degrees[1] != 0 else 0
            min = minutes[0] / minutes[1] if minutes[1] != 0 else 0
            sec = seconds[0] / seconds[1] if seconds[1] != 0 else 0
        else:
            deg = float(degrees)
            min = float(minutes)
            sec = float(seconds)
            if deg == 0 and min == 0 and sec == 0:
                return None

        decimal = deg + (min / 60.0) + (sec / 3600.0)

        if ref in ['S', 'W']:
            decimal = -decimal

        return decimal
    except Exception as e:
        print(f"Error parsing DMS: {e}")
        return None

def get_coordinates(geotags):
    if not geotags:
        return None, None
        
    lat = get_decimal_from_dms(geotags.get('GPSLatitude'), geotags.get('GPSLatitudeRef'))
    lon = get_decimal_from_dms(geotags.get('GPSLongitude'), geotags.get('GPSLongitudeRef'))

    return (lat, lon)

def get_make_model(exif):
    if not exif:
        return "Unknown_Make", "Unknown_Model"
        
    make = "Unknown_Make"
    model = "Unknown_Model"
    software = None
    for (idx, tag) in ExifTags.TAGS.items():
        if tag == 'Make' and idx in exif:
            make = str(exif[idx]).strip()
        elif tag == 'Model' and idx in exif:
            model = str(exif[idx]).strip()
        elif tag == 'Software' and idx in exif:
            software = str(exif[idx]).strip()
            
    if make == "Unknown_Make" and software:
        make = "Software_" + software[:20]
        model = "Unknown_Model"
        
    # Clean up strings for folder names
    make = "".join(c for c in make if c.isalnum() or c in (' ', '_', '-')).strip()
    model = "".join(c for c in model if c.isalnum() or c in (' ', '_', '-')).strip()
    
    if not make: make = "Unknown_Make"
    if not model: model = "Unknown_Model"
        
    return make, model

def is_inside_adamas(lat, lon):
    if lat is None or lon is None:
        return False
    return (ADAMAS_MIN_LAT <= lat <= ADAMAS_MAX_LAT) and (ADAMAS_MIN_LON <= lon <= ADAMAS_MAX_LON)

def get_campus_area(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), timeout=5)
        if location:
            address = location.raw.get('address', {})
            # Look for specific sub-areas, amenities, or roads
            area = address.get('amenity') or address.get('road') or address.get('suburb') or "Campus_General"
            area = "".join(c for c in str(area) if c.isalnum() or c in (' ', '_', '-')).strip()
            return area
    except GeocoderTimedOut:
        pass
    except Exception as e:
        print(f"Geocoding error: {e}")
        
    return "Campus_General"

def main():
    print("Starting Assignment 1 Processing...")
    
    # 1. Extract ZIP
    if not os.path.exists(EXTRACT_DIR):
        print(f"Extracting {ZIP_PATH}...")
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(EXTRACT_DIR)
        except Exception as e:
            print(f"Error extracting ZIP: {e}")
            return
            
    # 2. Setup Output Dir
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    report_data = []
    
    # 3. Process Images
    print("Processing images...")
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                continue
                
            filepath = os.path.join(root, file)
            
            try:
                # Need a fresh Image open to get EXIF, getexif doesn't need load but PIL might be finicky
                img = Image.open(filepath)
                exif = img.getexif()
                
                # GPS Info is stored in a separate IFD, getexif().get_ifd(0x8825) is more reliable in newer Pillow
                gps_ifd = exif.get_ifd(0x8825) if hasattr(exif, 'get_ifd') else {}
                
                geotags = {}
                for key, val in ExifTags.GPSTAGS.items():
                    if key in gps_ifd:
                        geotags[val] = gps_ifd[key]
                        
                # Fallback if get_ifd fails
                if not geotags:
                    geotags = get_geotagging(exif)

                make, model = get_make_model(exif)
                lat, lon = get_coordinates(geotags)
                
                img.close()
                
                location_category = "Unknown"
                campus_area = "N/A"
                
                if make == "Unknown_Make" and lat is None:
                    device_folder = "Unknown_Unclassified"
                    target_subfolder = ""
                else:
                    device_folder = f"{make}/{model}"
                    
                    if lat is not None and lon is not None:
                        if is_inside_adamas(lat, lon):
                            location_category = "Inside_Campus"
                            campus_area = get_campus_area(lat, lon)
                            target_subfolder = f"Inside_Campus/{campus_area}"
                        else:
                            location_category = "Outside_Campus"
                            target_subfolder = "Outside_Campus"
                    else:
                        location_category = "No_GPS"
                        target_subfolder = "No_GPS"
                
                # Determine target path
                if device_folder == "Unknown_Unclassified":
                    target_dir = os.path.join(OUTPUT_DIR, device_folder)
                else:
                    target_dir = os.path.join(OUTPUT_DIR, device_folder, target_subfolder)
                    
                os.makedirs(target_dir, exist_ok=True)
                
                target_filepath = os.path.join(target_dir, file)
                # Handle filename collisions
                counter = 1
                base, ext = os.path.splitext(file)
                while os.path.exists(target_filepath):
                    target_filepath = os.path.join(target_dir, f"{base}_{counter}{ext}")
                    counter += 1
                    
                shutil.copy2(filepath, target_filepath)
                
                report_data.append({
                    "Original_Filename": file,
                    "Make": make,
                    "Model": model,
                    "Latitude": lat if lat else "N/A",
                    "Longitude": lon if lon else "N/A",
                    "Location_Category": location_category,
                    "Campus_Area": campus_area,
                    "Segregated_Path": os.path.relpath(target_filepath, OUTPUT_DIR)
                })
                
            except Exception as e:
                print(f"Failed to process {file}: {e}")
                # Place in unknown
                target_dir = os.path.join(OUTPUT_DIR, "Unknown_Unclassified")
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(filepath, os.path.join(target_dir, file))
                report_data.append({
                    "Original_Filename": file,
                    "Make": "Error",
                    "Model": "Error",
                    "Latitude": "Error",
                    "Longitude": "Error",
                    "Location_Category": "Error",
                    "Campus_Area": "Error",
                    "Segregated_Path": os.path.relpath(os.path.join(target_dir, file), OUTPUT_DIR)
                })

    # 4. Generate Report
    print(f"Generating report at {REPORT_CSV}...")
    df = pd.DataFrame(report_data)
    df.to_csv(REPORT_CSV, index=False)
    
    # Also save as JSON
    report_json = REPORT_CSV.replace('.csv', '.json')
    print(f"Generating report at {report_json}...")
    df.to_json(report_json, orient='records', indent=4)
    
    # Cleanup temp directory
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
        
    print("Done!")

if __name__ == "__main__":
    main()
