{ A unit providing interfaces to BLAS routines }

unit numerik.blas;

{$mode objfpc}{$H+}

interface

uses
  dynlibs, multiarray;

const
{$IFDEF UNIX}
  {$IFDEF LINUX }
  BLAS_LIB_NAME = 'libopenblas.so';
  LAPACKE_LIB_NAME = 'liblapacke.so';
  {$ENDIF}
  {$IFDEF DARWIN}
  BLASLIB_NAME = 'libopenblas.dylib';
  {$ENDIF}
{$ENDIF}

{$IFDEF WINDOWS}
  BLAS_LIB_NAME = 'libopenblas.dll';
  LAPACKE_LIB_NAME = BLAS_LIB_NAME;
{$ENDIF}

  LAPACK_ROW_MAJOR = 101;
  LAPACK_COL_MAJOR = 102;
  CBLAS_ROW_MAJOR = 101;
  CBLAS_COL_MAJOR = 102;
  CBLAS_NO_TRANS = 111;
  CBLAS_TRANS = 112;
  CBLAS_CONJ_TRANS = 113;

type
  TCblasScopy = procedure(N: longint; X: TSingleVector; IncX: longint; Y:
    TSingleVector; IncY: longint); cdecl;
  TCblasSgemm = function(Order: longint; TransA: longint; TransB: longint;
    M: longint; N: longint; K: longint; alpha: single; A: TSingleVector;
    lda: longint; B: TSingleVector; ldb: longint; beta: single;
    C: TSingleVector; ldc: longint):longint; cdecl;
  TCblasSaxpy = procedure(N: longint; A: single; X: TSingleVector;
    INCX: longint; Y: TSingleVector; INCY: longint); cdecl;
  TLAPACKESgels = function(MatrixLayout: longint; Trans: char;
    M, N, NRHS: longint; A: TSingleVector; LDA: longint; B: TSingleVector;
    LDB: longint): longint; cdecl;
  TLAPACKESgeSVD = function(MatrixLayout: longint; JOBU, JOBVT: char;
    M, N: longint; A: TSingleVector; LDA: longint; S, U: TSingleVector;
    LDU: longint; VT: TSingleVector; LDVT: longint;
    Superb: TSingleVector): longint; cdecl;

var
  cblas_scopy: TCblasScopy;
  cblas_saxpy: TCblasSaxpy;
  cblas_sgemm: TCblassgemm;
  LAPACKE_sgels: TLAPACKESgels;
  LAPACKE_sgesvd: TLAPACKESgeSVD;
  BLASlibHandle: THandle = NilHandle;
  LAPACKElibHandle: THandle = NilHandle;

function Add_BLAS(A, B: TMultiArray): TMultiArray;
//function Add_Native(A, B: TMultiArray): TMultiArray;
function Sub_BLAS(A, B: TMultiArray): TMultiArray;
//function Sub_Native(A, B: TMultiArray): TMultiArray;
function MatMul_BLAS(A, B: TMultiArray): TMultiArray;
function MatMul_Native(A, B: TMultiArray): TMultiArray;

function MatMul_BACKEND(A, B: TMultiArray): TMultiArray;

function IsBLASAvailable: boolean;
function IsLAPACKEAvailable: boolean;

implementation

uses
  numerik;

function Add_BLAS(A, B: TMultiArray): TMultiArray;
begin
  Result := AllocateMultiArray(A.Size, False
    ).Reshape(A.Shape);
  Result.Data := nil;
  Result.Data := CopyVector(B.GetVirtualData);
  cblas_saxpy(A.Size, 1, A.GetVirtualData, 1, Result.Data, 1);
  Result.IsContiguous := True;
end;


function Sub_BLAS(A, B: TMultiArray): TMultiArray;
begin
  Result := AllocateMultiArray(A.Size, False
    ).Reshape(A.Shape);
  Result.Data := nil;
  Result.Data := CopyVector(A.GetVirtualData);
  cblas_saxpy(A.Size, -1, B.GetVirtualData, 1, Result.Data, 1);
end;


function MatMul_BLAS(A, B: TMultiArray): TMultiArray;
var
  OutSize: longint;
begin
  OutSize := A.Shape[0] * B.Shape[1];
  Result := AllocateMultiArray(OutSize).Reshape([A.Shape[0], B.Shape[1]]);

  cblas_sgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
    A.Shape[0], B.shape[1], B.shape[0], // A.shape[0], B.shape[1], B.shape[0]
    1,                                  // alpha
    A.Contiguous.Data, B.Shape[0],       // A, B.shape[0]
    B.Contiguous.Data, B.Shape[1],       // B, B.shape[1]
    1,                                  // beta
    Result.Data, B.Shape[1]             // C, B.shape[1]
    );
end;

function MatMul_Native(A, B: TMultiArray): TMultiArray;
var
  i, j, k: longint;
  sum: double;
begin
  Result := AllocateMultiArray(A.Shape[0] * B.Shape[1]).Reshape(
    [A.Shape[0], B.Shape[1]]);
  //SetLength(Result.Val, A.Shape[0] * B.Shape[1]);
  for i := 0 to A.shape[0] - 1 do
    for j := 0 to B.Shape[1] - 1 do
    begin
      sum := 0;
      for k := 0 to A.Shape[1] - 1 do
        sum := sum + A.Get(i * A.Shape[1] + k) * B.Get(k * B.Shape[1] + j);
      Result.Data[i * B.Shape[1] + j] := sum;
    end;
  //Result.Reshape([A.Shape[0], B.Shape[1]]);
end;

function MatMul_BACKEND(A, B: TMultiArray): TMultiArray;
begin
  if cblas_sgemm <> nil then
    Exit(MatMul_BLAS(A.Contiguous, B.Contiguous))
  else
  begin
    Exit(MatMul_Native(A.Contiguous, B.Contiguous));
  end;
end;

function IsBLASAvailable: boolean;
begin
  Exit(BLASlibHandle <> NilHandle);
end;

function IsLAPACKEAvailable: boolean;
begin
  Exit(LAPACKElibHandle <> NilHandle);
end;

initialization

  BLASlibHandle := LoadLibrary(BLAS_LIB_NAME);
  LAPACKElibHandle := LoadLibrary(LAPACKE_LIB_NAME);

  cblas_scopy := TCblasScopy(GetProcedureAddress(BLASlibHandle, 'cblas_scopy'));
  cblas_sgemm := TCblassgemm(GetProcedureAddress(BLASlibHandle, 'cblas_sgemm'));
  cblas_saxpy := TCblasSaxpy(GetProcedureAddress(BLASlibHandle, 'cblas_saxpy'));
  LAPACKE_sgels := TLAPACKESgels(GetProcedureAddress(LAPACKElibHandle,
    'LAPACKE_sgels'));
  LAPACKE_sgesvd := TLAPACKESgeSVD(GetProcedureAddress(LAPACKElibHandle,
    'LAPACKE_sgesvd'));

end.
