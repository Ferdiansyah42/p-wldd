import os
import random
import pandas as pd

def generate_sample_dataset(output_path="data/sample_poverty.csv", num_rows=1000):
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Candidate lists for realism
    first_names_male = ["Budi", "Joko", "Agus", "Bambang", "Herman", "Rudi", "Andi", "Hendra", "Slamet", "Dedi", "Ahmad", "Jaka", "Eko", "Wawan", "Tono"]
    first_names_female = ["Siti", "Sri", "Dewi", "Ani", "Maria", "Kartini", "Eka", "Ratna", "Rina", "Sarah", "Indah", "Wati", "Nur", "Utami", "Diana"]
    last_names = ["Susanto", "Prasetyo", "Santoso", "Setiawan", "Wijaya", "Kurniawan", "Sitorus", "Nasution", "Siregar", "Hidayat", "Saputra", "Wibowo", "Hadi", "Nugroho", "Purnama"]
    
    provinsi_kab = {
        "DKI JAKARTA": ["JAKARTA SELATAN", "JAKARTA TIMUR", "JAKARTA PUSAT", "JAKARTA UTARA", "JAKARTA BARAT"],
        "JAWA BARAT": ["BANDUNG", "BEKASI", "BOGOR", "DEPOK", "CIREBON", "GARUT", "TASIKMALAYA"],
        "JAWA TENGAH": ["SEMARANG", "SURAKARTA", "BANYUMAS", "BREBES", "CILACAP", "TEGAL"],
        "JAWA TIMUR": ["SURABAYA", "MALANG", "SIDOARJO", "GRESIK", "JEMBER", "BANYUWANGI", "BOJONEGORGO"],
        "BANTEN": ["TANGERANG", "SERANG", "CILEGON", "PANDEGLANG", "LEBAK"]
    }
    
    pekerjaan_list = ["Tani", "Buruh", "Pedagang", "Swasta", "PNS", "Tidak Bekerja"]
    pendidikan_list = ["SD", "SMP", "SMA", "S1"]
    sanitasi_list = ["Layak", "Tidak Layak"]
    air_minum_list = ["Layak", "Tidak Layak"]
    
    data = []
    
    random.seed(42)  # For reproducibility
    
    for _ in range(num_rows):
        # 1. Identity Columns
        nik_province_code = random.choice([31, 32, 33, 35, 36])
        nik_regency_code = random.randint(10, 90)
        nik_birth_date = f"{random.randint(1, 28):02d}{random.randint(1, 12):02d}{random.randint(40, 99):02d}"
        nik_seq = f"{random.randint(1, 9999):04d}"
        nik = f"{nik_province_code}{nik_regency_code}01{nik_birth_date}{nik_seq}"
        
        is_male = random.choice([True, False])
        first = random.choice(first_names_male) if is_male else random.choice(first_names_female)
        last = random.choice(last_names)
        nama = f"{first} {last}"
        
        # 2. Regional Columns
        prov = random.choice(list(provinsi_kab.keys()))
        kab = random.choice(provinsi_kab[prov])
        
        # 3. Socio-economic Features
        umur = random.randint(18, 80)
        pekerjaan = random.choice(pekerjaan_list)
        pendidikan = random.choice(pendidikan_list)
        tanggungan = random.randint(1, 6)
        
        # Pendapatan correlates with Pekerjaan and Pendidikan
        if pekerjaan == "PNS":
            pendapatan = random.randint(4000000, 10000000)
        elif pekerjaan == "Swasta":
            pendapatan = random.randint(3000000, 8000000)
        elif pekerjaan == "Pedagang":
            pendapatan = random.randint(1500000, 6000000)
        elif pekerjaan in ["Buruh", "Tani"]:
            pendapatan = random.randint(800000, 2500000)
        else: # Tidak Bekerja
            pendapatan = random.randint(0, 1000000)
            
        # Pendidikan adjustment
        if pendidikan == "S1":
            pendapatan = int(pendapatan * 1.5)
        elif pendidikan == "SD":
            pendapatan = int(pendapatan * 0.8)
            
        # Housing and amenities
        # Smaller house for lower income
        if pendapatan < 2000000:
            luas_rumah = random.randint(9, 45)
            sanitasi = random.choices(sanitasi_list, weights=[0.4, 0.6])[0]
            air_minum = random.choices(air_minum_list, weights=[0.4, 0.6])[0]
        else:
            luas_rumah = random.randint(36, 150)
            sanitasi = random.choices(sanitasi_list, weights=[0.9, 0.1])[0]
            air_minum = random.choices(air_minum_list, weights=[0.9, 0.1])[0]
            
        # 4. Target Variable: Status Kelayakan (Bansos Eligibility)
        # Low income, high dependents, poor facilities make it more eligible
        score = 0.0
        # Income factor (lower is more eligible)
        score += (3000000 - pendapatan) / 1000000
        # Dependents factor
        score += (tanggungan - 2) * 0.5
        # House area factor (smaller is more eligible)
        score += (45 - luas_rumah) / 15
        # Pekerjaan factor
        if pekerjaan in ["Tidak Bekerja", "Buruh", "Tani"]:
            score += 1.2
        elif pekerjaan == "PNS":
            score -= 2.0
            
        # Amenities factor
        if sanitasi == "Tidak Layak":
            score += 0.8
        if air_minum == "Tidak Layak":
            score += 0.8
            
        # Threshold for eligibility
        is_eligible = score > 0.3
        
        # Add 5% noise/flip to simulate real-world data complexity
        if random.random() < 0.05:
            is_eligible = not is_eligible
            
        status_kelayakan = "Layak" if is_eligible else "Tidak Layak"
        
        data.append({
            "NIK": nik,
            "Nama": nama,
            "Provinsi": prov,
            "Kab/Kota": kab,
            "Umur": umur,
            "Pekerjaan": pekerjaan,
            "Pendidikan": pendidikan,
            "Pendapatan": pendapatan,
            "Tanggungan": tanggungan,
            "Luas Rumah": luas_rumah,
            "Sanitasi": sanitasi,
            "Air Minum": air_minum,
            "Status Kelayakan": status_kelayakan
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_rows} rows at {output_path}")
    return df

if __name__ == "__main__":
    generate_sample_dataset()
