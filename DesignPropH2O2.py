import numpy as np
import matplotlib.pyplot as plt
import cmath  # Módulo para manejar números complejos


### Ok
def MassEstimation(F_W, m_ini):
  F     = F_W * go * m_ini
  mE    = F/(go*(0.0006098*F+13.44))
  LE    =  (0.0054*F+31.92)/100
  DE    = (0.00357*F + 14.48)/100
  F_W_final  = F/(mE*go)
  return F, mE, LE, DE, F_W_final

def CombustionParameters(Propellant, OF):

  # Añadir información de más curvas para distintos tipos de propelentes
  if Propellant == "O2/H2":
    
    if OF == 2.6:
      TF, y, M = 2250, 1.265, 7.5

    elif OF == 2.8:
      TF, y, M = 2350, 1.26, 7.8

    elif OF == 3.8:
      TF, y, M = 2800, 1.225, 9.7

    elif OF == 4.6:
      TF, y, M = 3200, 1.215, 11
    
    elif OF == 5:
      TF, y, M = 3250, 1.21, 11.8
    
    elif OF == 5.2:
      TF, y, M = 3300, 1.21, 12

    elif OF == 5.2:
      TF, y, M = 3300, 1.21, 12

    elif OF == 6:
      TF, y, M = 3400, 1.2, 13

    elif Propellant == "O2/RP-1":
      OF  = 2.3 
      TF  = 3510
      y   = 1.225
      M   = 22.1
  return OF, TF, y, M

def SpecificImpulse(y,nc,Tc,M,pc,pa,lamb,FigName='',label=''):
  R       = 8314/M
  c_x     = nc * np.sqrt(y*R*Tc) / ( y * (2/(y+1))**((y+1)/(2*y-2)) )

  Mach    = np.arange(1,6,0.0001)
  k = y
  epsilon = 1/Mach*((2/(k+1))*(1+(k-1)/2*Mach**2))**((k+1)/(2*k-2))
  pe      = pc * (1+((k-1)/2)*Mach**2)**(k/(1-k))

  a = c_x * y /go
  b = (2/(y-1))*(2/(y+1))**((y+1)/(y-1))*(1-(pe/pc)**((y-1)/y))
  c = c_x*epsilon*(pe-pa)/(go*pc)
  Isp = lamb*(a*np.sqrt(b)+c)*1

  plt.figure(f"{FigName}")
  plt.title("Specific Impulse vs Nozzle Expansion Rate")
  plt.plot(epsilon, Isp, label=f"{label}")
  plt.xlabel("Nozzle Expansion (rate)")
  plt.ylabel("Specific Impulse (s)")
  plt.legend()
  plt.grid('on')
  
  return Isp, epsilon, c_x

def PressureParameters(pc,rho,v,NamePropellant):
  dP_injector = 0.2*pc
  dP_feed   = 50e3
  dP_dynamic  = 0.5 * rho * v**2
  P_tank    = pc + dP_dynamic + dP_feed + dP_injector
  dP_cool     = 0.15*pc
  return dP_injector, dP_feed, dP_dynamic, P_tank

def PropellantMass(m_payload, dV, Isp, go, f_inert):
  m_prop = m_payload * ( np.exp(dV/(Isp*go))-1 ) * (1-f_inert) / (1 - f_inert*np.exp(dV/(Isp*go)))
  return m_prop

def PointYX(y,x,xo,NamePropellant):
  index_xo = np.argmin(np.abs(x - xo))
  xo_aprox = x[index_xo]
  yo_aprox = y[index_xo]
  return xo_aprox, yo_aprox

def InertMass(f_inert,m_total):
  m_inert = m_prop*f_inert/(1-f_inert)
  return m_inert

def TotalImpulse(Isp,m_prop,go):
  I = Isp*m_prop*go
  return I

def BurnTime(I,F):
  t_b = I/F
  return t_b

def FuelOxidizerMassFlow(m_prop, OF, t_b):
  m_fuel = m_prop/(1+OF)
  dot_m_fuel = m_fuel/t_b

  m_ox = m_prop*OF/(1+OF)
  dot_m_ox = m_ox/t_b

  dot_m_prop = dot_m_fuel + dot_m_ox

  return m_fuel, dot_m_fuel, m_ox, dot_m_ox, dot_m_prop

def FuelOxidizerVolumenRadio(m_fuel, m_ox, rho_fuel, rho_ox):
  V_fuel = m_fuel/rho_fuel
  r_fuel = (V_fuel*3/(4*np.pi))**(1/3)

  V_ox   = m_ox/rho_ox
  r_ox   = (V_ox*3/(4*np.pi))**(1/3)
  return V_fuel, r_fuel, V_ox, r_ox 

def ThroatArea(c_x,dot_m_prop,pc):
  At = dot_m_prop*c_x/pc 
  return At

def CrossSectionalDiameter(Area):
  D = 2*np.sqrt(Area/np.pi)
  return D

def CombustionChamberCrossSectionalAverageArea(At,Mach,y):
  Ac = At*1/Mach*((2/(y+1))*(1+(y-1)/2*Mach**2))**((y+1)/(2*(y-1)))
  return Ac

def CombustionChamberLength(Lx,At,Ac):
  L = Lx*At/Ac
  return L

def CombustionChamberWallThickness(pc,fsafe,Dc,Ftu,fmass):
  pb = pc*fsafe
  tw = pb*(Dc/2)/Ftu*fmass
  return tw

def CombustionChamberMass(rho_chamber,tw,Lcc,rcc,rt,theta_cc):
  mcc = np.pi*rho_chamber*tw*(2*rcc*Lcc+np.pi*(rcc**2-rt**2))/np.tan(theta_cc)
  return mcc

def ConicalNozzleLength(De,Dt,theta_cn):
  Ln = (De-Dt)/(2*np.tan(theta_cn))
  return Ln

def NozzleConstantThicknessMass(rho_chamber,tw,Ln,re,rt):
  mn = np.pi*rho_chamber*tw*Ln*(re+rt)
  return mn

def NozzleTaperedWallMass(rho_chamber,tt,te,Ln,re,rt):
  f1 = (te-tt)/Ln
  f2 = (re-rt)/Ln
  mn = 2*np.pi*rho_chamber*Ln*(1/3*f1*f2*Ln**2+1/2*(f1*rt+f2*tt)*Ln+rt*tt)
  return mn

def EngineMass(RefMass, f_engine, f_injector, f_ablative):
  m_engine = f_engine * RefMass
  m_injector = f_injector * m_engine
  m_ablativematerial = f_ablative * m_engine 
  return m_engine, m_injector, m_ablativematerial

def TankMass(pb,V_tank,phi_tank,go,fsafe):
  m_tank = fsafe*pb*V_tank/(phi_tank*go)
  return m_tank

def PressurizerSelection(PressurizerName):
  
  if PressurizerName == "He":
    y = 1.66
    MassMol = 4.003

  return y, MassMol 

def FinalTemperature(Ti,Pi,Pf,y):
  Tf = Ti * (Pf/Pi) ** ((y-1)/y)
  return Tf

def PressurizerMassVolumeEstimation(Pi, Ti, Pf, Tf, f_design, V_pressurizer_i, MassMol, Ru):
  m_pressurizer_f = (f_design*Pf) * V_pressurizer_i * MassMol /(Ru * Tf)
  V_pressurizer_f = m_pressurizer_f * Ru * Ti / (Pi * MassMol)
  return m_pressurizer_f, V_pressurizer_f

def TVCMassEstimation(mE,m_engine):
  m_TVC = mE - m_engine
  return m_TVC

def StructuralMountsMassEstimation(mass,f_mass):
  m_structuralmounts = mass * f_mass
  return m_structuralmounts

def TorusRadius(V,L):
    a, b, c, d = 2*np.pi**2, -L*np.pi**2, 0, V
    coef = [a, b, c, d]
    raices = np.roots(coef)
    raices = np.real(raices)
    radio_min = np.min(raices[raices>0])
    ID = L-4*radio_min
    return radio_min, ID






#########  5.5.2 Decisiones preliminares de Diseño

### Contantes y requerimientos

Margen    = 0.1             # rate
dV        = 1721*(1+Margen)            # m/2
m_ini     = 12000           # kg
F_W       = 0.3             # rate
m_payload = 4914            # kg

### Otros
rn        = 3
go        = 9.80665

### 1. Estimar la masa y la envolvente del sistema
F, mE, LE, DE, F_W_final = MassEstimation(F_W, m_ini)

print("\n\n")
print(f"Delta de velocidad de diseño  ΔV   = {round(dV,rn)} m/s")
print(f"Fuerza de empuje requerido    F    = {round(F/1000,rn)} KN")
print(f"Masa del motor                mE   = {round(mE,rn)} Kg")
print(f"Longitud del motor            LE   = {round(LE,rn)} m")
print(f"Diámetro del motor            DE   = {round(DE,rn)} m")
print(f"Relación Empuje-Peso          F/W = {round(F_W_final,rn)} ")



### 2. Selección de los propelentes
FuelName1          = "H2"
OxidizerName       = "O2"

print(f"\n\nPropenlentes seleccionado: {FuelName1},{OxidizerName}")



### 3. Determinación del ciclo del motor y enfoque de enfriamiento
# Tanques presurizados por tanque de presión y motor tipo ablativo




### 4. Determinación de lo niveles de presión para el motor y el sistema de alimentación
OF, TF_fuel, y_fuel, MassMol_fuel = CombustionParameters(f"{OxidizerName}/{FuelName1}",6)

nc    = 1
pc    = 7e5
epsilon_o = 150
pa    = 0
lamb  = 0.98
Isp_fuel, epsilon_fuel, cx = SpecificImpulse(y_fuel,nc,TF_fuel,MassMol_fuel,pc,pa,lamb,"Isp vs epsilon", f"{FuelName1}/{OxidizerName}")

rho_fuel 		 = 71#71
rho_oxidizer = 1142
v 			     = 10

dP_injector, dP_feed, dP_dynamic_fuel, P_tank_fuel = PressureParameters(pc, rho_fuel, v, FuelName1)
dP_injector, dP_feed, dP_dynamic_oxidizer , P_tank_oxidizer = PressureParameters(pc, rho_oxidizer , v, OxidizerName)

print(f"\n\nRelación de mezcla      O/F           = {round(OF,rn)} rate")
print(f"Temperatura de flama        TF_fuel       = {round(TF_fuel,rn)} K")
print(f"Parámetro isentrópico       y_fuel        = {round(y_fuel,rn)} rate")
print(f"Masa Molecular              MassMol_fuel  = {round(MassMol_fuel,rn)} Kg/Kmol, g/mol")

print(f"\n\nRelación de expansión                    epsilon_o = {round(epsilon_o,rn)} rate")
print(f"Presión de la cámara de combustión           pc        = {round(pc/1000,rn)} kPa")

print(f"\nCaida de presión en el inyector                    ΔP_injector     = {round(dP_injector/1000,rn)} kPa")
print(f"Caida de presión en el sistema de alimentación      ΔP_feed         = {round(dP_feed/1000,rn)} kPa")
print(f"Presión dinámica del {FuelName1}                    ΔP_propellant   = {round(dP_dynamic_fuel/1000,rn)} kPa")
print(f"Presión del tanque de {FuelName1}                   P_tank_fuel     = {round(P_tank_fuel/1000,rn)} kPa")
print(f"Presión dinámica del {OxidizerName}                 ΔP_propellant   = {round(dP_dynamic_oxidizer/1000,rn)} kPa")
print(f"Presión del tanque de {OxidizerName}                P_tank_oxidizer = {round(P_tank_oxidizer/1000,rn)} kPa")



### 5. Estimar la masa del propelente y el tamaño de sus tanques

f_inert   = 0.261543889

epsilon_o_aprox_fuel, Isp_o_fuel = PointYX(Isp_fuel, epsilon_fuel, epsilon_o,  f"{FuelName1}/{OxidizerName}")

m_prop  = PropellantMass(m_payload, dV, Isp_o_fuel, go, f_inert)

m_inert = InertMass(f_inert,m_prop)
I       = TotalImpulse(Isp_o_fuel,m_prop,go)
t_b     = BurnTime(I,F)

m_fuel, dot_m_fuel, m_ox, dot_m_ox, dot_m_prop = FuelOxidizerMassFlow(m_prop, OF, t_b)
V_fuel, r_fuel, V_ox, r_ox = FuelOxidizerVolumenRadio(m_fuel, m_ox, rho_fuel, rho_oxidizer)

print(f"\n\nMasa de la carga útil,                              m_payload    = {round(m_payload,rn)} kg") 
print(f"Fracción de masa estructural                        f_inert      = {round(f_inert,rn)} ")
print(f"Masas inicial del vehículo                          m_ini        = {round(m_ini,rn)} kg")

print(f"\nImpulso Específico                                  Isp          = {round(Isp_o_fuel,rn)} s")
print(f"Masa del propelente                                 m_prop       = {round(m_prop,rn)} kg")
print(f"Masa inerte de la etapa                             m_inert      = {round(m_inert,rn)} kg")
print(f"Nivel de Empuje                                     F            = {round(F,rn)} N")
print(f"Impulsto Total                                      I            = {round(I,rn)} Ns")
print(f"Tiempo de quemadura                                 t_b          = {round(t_b,rn)} s")

print(f"\nCaudal del propelente                               dot_m_prop   = {round(dot_m_prop,rn)} kg/s")
print(f"Caudal del combustible  {FuelName1}                 dot_m_fuel   = {round(dot_m_fuel,rn)} kg/s")
print(f"Caudal del oxidante  {OxidizerName}                 dot_m_ox     = {round(dot_m_ox,rn)} kg/s")      

print(f"\nMasa del combustible  {FuelName1}                   m_fuel       = {round(m_fuel,rn)} kg")
print(f"Masa del oxidante  {OxidizerName}                   m_ox         = {round(m_ox,rn)} kg")

print(f"\nDensidad del combustible {FuelName1}                rho_fuel       = {round(rho_fuel,rn)} kg/m^3")
print(f"Densidad del oxidante {OxidizerName}                rho_ox         = {round(rho_oxidizer,rn)} kg/m^3")

print(f"\nVolumen del tanque de combustible  {FuelName1}      V_fuel       = {round(V_fuel,rn)} m^3")
print(f"Volumen del tanque de oxidante  {OxidizerName}      V_ox         = {round(V_ox,rn)} m^3")

print(f"\nRadio del tanque de combustible  {FuelName1}        r_fuel       = {round(r_fuel,rn)} m")
print(f"Radio del tanque de oxidante  {OxidizerName}        r_ox         = {round(r_ox,rn)} m")


#########  5.5.3 Dimensionamiento, Diseño y Compensaciones del Sistema

### 1. Cámara de empuje
At    = ThroatArea(cx,dot_m_prop,pc)
Ae    = epsilon_o * At

Dt    = CrossSectionalDiameter(At)
De    = CrossSectionalDiameter(Ae)

Mach  = 0.2
Lx    = 0.9
Ac    = CombustionChamberCrossSectionalAverageArea(At,Mach,y_fuel)
L     = CombustionChamberLength(Lx,At,Ac)
Dc    = CrossSectionalDiameter(Ac)
ContractionRatio = Ac/At

fsafe = 2
fmass = 1.5
Fty   = 310e6         # Esfuerzo en el límite elástico del niobio
tw    = CombustionChamberWallThickness(pc,fsafe,Dc,Fty,fmass)

rho_chamber = 8500              # Densidad del material del motor, niobio o columbium
theta_cc    = 45*np.pi/180         # Ángulo medio de constricción constante
mcc         = CombustionChamberMass(rho_chamber,tw,L,Dc/2,Dt/2,theta_cc)

theta_cn = 15*np.pi/180         # Ángulo medio del cono de la tobera
Ln        = ConicalNozzleLength(De,Dt,theta_cn)

Lf    = 0.675                      # DETERMINADO POR FIGURA LAMBDA VS Lf @epsilon
Lcn   = Lf * Ln
#mn   = NozzleConstantThicknessMass(rho_chamber,tw,Ln,De/2,Dt/2)
mn    = NozzleTaperedWallMass(rho_chamber,tw,tw/10,Lcn,De/2,Dt/2)
m_ref_historic       = 163
f_mass_engine        = 0.433374231
f_mass_injector      = 0.249
f_mass_ablative      = 0.352
#m_engine, m_injector, m_ablativematerial = EngineMass(163,0.433374231,0.249,0.352)
m_engine, m_injector, m_ablativematerial = EngineMass(m_ref_historic, f_mass_engine, f_mass_injector, f_mass_ablative)

print(f"\n\nVelocidad característica de escape    c* = {round(cx,rn)} m/s") 
print(f"Área en la garganta                   At = {round(At,rn)} m^2")
print(f"Área a la salida                      Ae = {round(Ae,rn)} m^2")
print(f"Diámetro en la garganta               Dt = {round(Dt,rn)} m^2")
print(f"Diámetro a la salida                  De = {round(De,rn)} m^2") 

print(f"\nLongitud en la cámara                 L = {round(L,rn)} m")
print(f"Área en la cámara                     Ac = {round(Ac,rn)} m^2")
print(f"Diámetro en la cámara                 Dc = {round(Dc,rn)} m")
print(f"Índice de contracción                 ContractionRatio = {round(ContractionRatio,rn)}")

print(f"\nEspesor de pared de la cámara         tw = {round(tw,rn)} m")
print(f"Masa de la cámara                     mcc = {round(mcc,rn)} kg")
print(f"Longitud de la tobera                 Ln = {round(Ln,rn)} m @{round(theta_cn*180/np.pi)}°")
print(f"Fracción de tobera cónica             Lf = {round(Lf,rn)}   @{round(theta_cn*180/np.pi)}°")
print(f"Longitud de campana de la tobera      Lcn = {round(Lcn,rn)} m")
print(f"Masa de la tobera                     mn = {round(mn,rn)} kg")

print(f"\nMasa total del motor                  m_engine = {round(m_engine,rn)} kg")
print(f"Masa del inyector                     m_injector = {round(m_injector,rn)} kg")
print(f"Masa del Material ablativo            m_ablativematerial = {round(m_ablativematerial,rn)} kg")



### 2. Determinación del ciclo del motor y enfoque de enfriamiento
# Tanques presurizados por tanque de presión y motor tipo ablativo



### 3. Sistema de almacenamiento de propelente
phi_tank = 2500
m_tank_fuel = TankMass(P_tank_fuel,V_fuel,phi_tank,go,fsafe)
m_tank_oxidizer = TankMass(P_tank_oxidizer,V_ox,phi_tank,go,fsafe)

print(f"\n\nMasa del tanque {FuelName1}               m_tank_fuel     = {round(m_tank_fuel,rn)} kg")
print(f"Masa del tanque {OxidizerName}            m_tank_oxidizer = {round(m_tank_oxidizer,rn)} kg")



### 4. Diseño del sistema de presurización del tanque
PressurizerName = "He"
y_pressurizer, MassMol_pressurizer = PressurizerSelection(PressurizerName)

Ti_pressurizer = 273
Pi_pressurizer = 21e6
Pf_pressurizer = 938800
Tf_pressurizer = FinalTemperature(Ti_pressurizer,Pi_pressurizer,Pf_pressurizer,y_pressurizer)

Ru                    = 8314
f_design              = 1.05
V_pressurizer_initial = V_fuel + V_ox
m_pressurizer, V_pressurizer = PressurizerMassVolumeEstimation(Pi_pressurizer, Ti_pressurizer, Pf_pressurizer, Tf_pressurizer, f_design, V_pressurizer_initial,MassMol_pressurizer,Ru)

phi_tank_pressurizer = 6350
m_tank_pressurizer   = TankMass(Pi_pressurizer,V_pressurizer,phi_tank_pressurizer,go,1)

print(f"\n\nParámetros isentrópico {PressurizerName}      y_pressurizer       = {round(y_pressurizer,rn)} ")
print(f"Masa molecular {PressurizerName}                  MassMol_pressurizer = {round(MassMol_pressurizer,rn)} kg/Kmol")

print(f"\nTemperatura final del tanque de presurizante {PressurizerName}     Tf_pressurizer      = {round(Tf_pressurizer,rn)} K")

print(f"\nMasa del presurizante {PressurizerName}                            m_pressurizer       = {round(m_pressurizer,rn)} kg")
print(f"Volumen del tanque del presurizante {PressurizerName}               V_pressurizer       = {round(V_pressurizer,rn)} m^3")
print(f"Masa del tanque del presurizante {PressurizerName}                  m_tank_pressurizer  = {round(m_tank_pressurizer,rn)} kg")



### 5. Control vectorial de Empuje
m_TVC = TVCMassEstimation(mE,m_engine)

print(f"\nMasa del control vectorial de empuje                    m_TVC = {round(m_TVC,rn)} kg")



### 6. Montaje estructural
mass   = m_engine +  m_tank_pressurizer + m_tank_fuel + m_tank_oxidizer
f_mass = 0.1
m_structuralmounts = StructuralMountsMassEstimation(mass,f_mass)

print(f"\nMasa del montaje estructural del presurizante             m_structuralmounts = {round(m_structuralmounts,rn)} kg")



#########  5.5.4 Diseño de la linea base
L = 3
radio_tank_torus_ox, ID_tank_torus_ox                   = TorusRadius(V_ox,L)
radio_tank_torus_fuel, ID_tank_torus_fuel               = TorusRadius(V_fuel,L)
radio_tank_torus_pressurizer, ID_tank_torus_pressurizer = TorusRadius(V_pressurizer,L)

print(f"\n\nRadio del toriode del tanque de oxidante {OxidizerName}       radio_tank_torus_ox   = {round(radio_tank_torus_ox,rn)} m")
print(f"Diámetro interior del tanque de oxidante {OxidizerName}       ID_tank_torus_ox      = {round(ID_tank_torus_ox,rn)} m")
print(f"\nRadio del toriode del tanque de combustible {FuelName1}       radio_tank_torus_fuel = {round(radio_tank_torus_fuel,rn)} m")
print(f"Diámetro interior del tanque de combustible {FuelName1}       ID_tank_torus_fuel    = {round(ID_tank_torus_fuel,rn)} m")
print(f"\nRadio del toriode del tanque presurizante {PressurizerName}       radio_tank_torus_pressurizer = {round(radio_tank_torus_pressurizer,rn)} m")
print(f"Diámetro interior del tanque presurizante {PressurizerName}       ID_tank_torus_pressurizer    = {round(ID_tank_torus_pressurizer,rn)} m")
m_total = m_prop + m_inert + m_payload + m_tank_fuel + m_tank_oxidizer+ m_tank_pressurizer + m_engine + m_injector + m_ablativematerial + m_TVC + m_structuralmounts
print(f"m_total       m_total    = {round(m_total,rn)} m")
plt.show()

