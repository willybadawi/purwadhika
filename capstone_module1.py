datakaryawan = [
    {'nik': 12345, 'nama': 'Angga', 'jabatan': 'Staff', 'gaji': 8000000, 'divisi': 'Sales'},
    {'nik': 23456, 'nama': 'Budi', 'jabatan': 'Staff', 'gaji': 10000000, 'divisi': 'Finance'},
    {'nik': 34567, 'nama': 'Cheryl', 'jabatan': 'Supervisor', 'gaji': 15000000, 'divisi': 'Sales'},
    {'nik': 45678, 'nama': 'Dharma', 'jabatan': 'Supervisor', 'gaji': 18000000, 'divisi': 'Finance'},
    {'nik': 56789, 'nama': 'Eko', 'jabatan': 'Manager', 'gaji': 25000000, 'divisi': 'Sales'},
    {'nik': 67899, 'nama': 'Fany', 'jabatan': 'Manager', 'gaji': 28000000, 'divisi': 'Finance'}
]

def data_karyawan():
    print('\t\t\t\tDATA KARYAWAN PT.ABC')
    print('\t\t\t\t--------------------')
    print('INDEX\t| NIK\t| NAMA\t\t| JABATAN\t| GAJI\t\t| DIVISI')
    
    for i, karyawan in enumerate(datakaryawan):
        print(f'{i}\t| {karyawan["nik"]}\t| {karyawan["nama"].ljust(10)}\t| {karyawan["jabatan"].ljust(10)}\t| {karyawan["gaji"]}\t| {karyawan["divisi"]}')

def read_menu():
    while True:
        pilihanMenu = input('''
        #####################################
        List Menu :
        1. Menampilkan Semua Data Karyawan
        2. Menampilkan data berdasarkan NIK
        3. Kembali ke menu Utama
        #####################################
        Masukkan pilihan (1-3): ''')

        if pilihanMenu == '1':
            data_karyawan()
        elif pilihanMenu == '2':
            menampilkan_data_nik()
        elif pilihanMenu == '3':
            break
        else:
            print('Pilihan tidak ada, silakan coba lagi.')

def menampilkan_data_nik():
    while True:
        try:
            input_nik = int(input('Masukkan NIK yang ingin Anda cari: '))
            break 
        except ValueError:
            print("NIK harus berupa angka. Silakan coba lagi.")
    
    print('Hasil pencarian berdasarkan NIK:')
    for karyawan in datakaryawan:
        if input_nik == karyawan['nik']:
            print('NIK\t| NAMA\t\t| JABATAN\t| GAJI\t\t| DIVISI')
            print(f'{karyawan["nik"]}\t| {karyawan["nama"].ljust(10)}\t| {karyawan["jabatan"].ljust(10)}\t| {karyawan["gaji"]}\t| {karyawan["divisi"]}')
            return  # Keluar dari fungsi jika data ditemukan
    
    print(f'Tidak ada karyawan dengan NIK = {input_nik}.')

############################### Create Menu ######################################
def create_menu():
    while True:
        pilihanMenu = input('''
        #####################################
        Menu Menambahkan Data:
        1. Tambah Data Karyawan Baru
        2. Kembali ke menu Utama
        #####################################
        Masukkan pilihan (1-2): ''')

        if pilihanMenu == '1':
            menambah_data_karyawan_baru()
        elif pilihanMenu == '2':
            break
        else:
            print('Pilihan tidak ada, silakan coba lagi.')

def menambah_data_karyawan_baru():
    while True:
        nik = input('Masukkan NIK (hanya angka): ')
        if nik.isdigit():  # Memeriksa apakah NIK hanya berisi angka
            nik = int(nik)
            if any(k['nik'] == nik for k in datakaryawan):
                print(f'NIK {nik} sudah terdaftar!')
                continue
            else:
                break
        else:
            print("NIK yang anda masukan bukan angka! Harap masukan NIK yang benar.")

    while True:
        nama = input('Masukkan Nama: ')
        if nama.strip():  # Memeriksa apakah nama tidak kosong (termasuk spasi)
            break
        else:
            print("Nama tidak boleh kosong! Harap masukan nama yang benar.")
            
    while True:
        jabatan = input('Masukkan Jabatan: ')
        if jabatan.strip():  # Memeriksa apakah jabatan tidak kosong (termasuk spasi)
            break
        else:
            print("Jabatan tidak boleh kosong! Harap masukan jabatan yang benar.")

    while True:
        gaji = input('Masukkan Gaji: ')
        if gaji.isdigit():  # Memeriksa apakah Gaji hanya berisi angka
            gaji = int(gaji)
            break
        else:
            print("Gaji harus berupa angka! Harap masukan Gaji dengan benar.")

    while True:
        divisi = input('Masukkan Divisi: ')
        if divisi.strip():  # Memeriksa apakah divisi tidak kosong (termasuk spasi)
            break
        else:
            print("Divisi tidak boleh kosong! Harap masukan divisi yang benar.")

    datakaryawan.append({
        'nik': nik,
        'nama': nama,
        'jabatan': jabatan,
        'gaji': gaji,
        'divisi': divisi
    })
    print(f'Data karyawan bernama {nama} berhasil ditambahkan!')

############################### Update Menu ######################################
def update_menu():
    while True:
        pilihanMenu = input('''
        #####################################
        List Menu :
        1. Update data karyawan
        2. Kembali ke menu Utama
        #####################################
        Masukkan pilihan (1-2): ''')

        if pilihanMenu == '1':
            update_data_karyawan()
        elif pilihanMenu == '2':
            break
        else:
            print('Pilihan tidak ada, silakan coba lagi.')

def update_data_karyawan():
    data_karyawan()
    while True:
        try:
            nik_karyawan = int(input('Masukkan NIK karyawan yang ingin diupdate: '))
            found = False
            for karyawan in datakaryawan:
                if nik_karyawan == karyawan['nik']:
                    found = True
                    karyawan['jabatan'] = input('Masukkan Jabatan Baru: ')
                    while True:
                        try:
                            karyawan['gaji'] = int(input('Masukkan Gaji Baru: '))
                            break
                        except ValueError:
                            print("Gaji harus berupa angka! Harap masukkan Gaji yang benar.")
                    karyawan['divisi'] = input('Masukkan Divisi Baru: ')
                    print(f"Data Karyawan dengan NIK {karyawan['nik']}, bernama {karyawan['nama']} berhasil diperbarui.")
                    return
            if not found:
                print(f'Karyawan dengan NIK {nik_karyawan} tidak ditemukan.')
                break
        except ValueError:
            print("Input NIK harus berupa angka. Silakan coba lagi.")

############################### Delete Menu ######################################
def delete_menu():
    while True:
        pilihanMenu = input('''
        #####################################
        List Menu :
        1. Menghapus data karyawan
        2. Kembali ke menu Utama
        #####################################
        Masukkan pilihan (1-2): ''')

        if pilihanMenu == '1':
            menghapus_data_karyawan()
        elif pilihanMenu == '2':
            break
        else:
            print('Pilihan tidak ada, silakan coba lagi.')

def menghapus_data_karyawan():
    data_karyawan()
    while True:
        try:
            index = int(input('Masukkan index karyawan yang ingin dihapus (atau ketik -1 untuk batal): '))
            if index == -1:
                print('Penghapusan data dibatalkan.')
                break
            if 0 <= index < len(datakaryawan):
                checker = input(f'Apakah Anda yakin data karyawan bernama {datakaryawan[index]["nama"]} akan dihapus? (ya/tidak): ')
                if checker.lower() != 'ya':
                    print('Penghapusan data dibatalkan.')
                    break
                removed = datakaryawan.pop(index)
                print(f'Data karyawan dengan nama {removed["nama"]} berhasil dihapus.')
                break
            else:
                print('Indeks tidak valid. Masukkan indeks yang benar.')
        except ValueError:
            print('Input harus berupa angka. Silakan coba lagi.')

############################### Main Menu ######################################
while True:
    pilihanMenu = input('''
                    Selamat Datang 
            di Aplikasi Data Karyawan PT. ABC
        #####################################
        List Menu :
        1. Menampilkan Data Karyawan
        2. Menambah Data Karyawan
        3. Mengupdate Data Karyawan
        4. Menghapus Data Karyawan
        5. Exit Program
        #####################################
        Masukkan pilihan (1-5): ''')
    if pilihanMenu == '1':
        read_menu()
    elif pilihanMenu == '2':
        create_menu()
    elif pilihanMenu == '3':
        update_menu()
    elif pilihanMenu == '4':
        delete_menu()  
    elif pilihanMenu == '5':
        print('Terima kasih telah menggunakan Aplikasi Data Karyawan PT.ABC!')
        break
    else:
        print('Pilihan tidak ada, silakan coba lagi.')
