{ A unit providing interfaces to BLAS routines }

unit numerik.blas;

{$mode objfpc}{$H+}

interface

uses
  multiarray;

const

  LIB_NAME = 'libopenblas.dll';

  LAPACK_ROW_MAJOR = 101;
  LAPACK_COL_MAJOR = 102;
  CBLAS_ROW_MAJOR = 101;
  CBLAS_COL_MAJOR = 102;
  CBLAS_NO_TRANS = 111;
  CBLAS_TRANS = 112;
  CBLAS_CONJ_TRANS = 113;

function Add_BLAS(A, B: TMultiArray): TMultiArray;
function MatMul_BLAS(A, B: TMultiArray): TMultiArray;

procedure cblas_saxpy(N: longint; A: single; X: TSingleVector; INCX: longint;
  Y: TSingleVector; INCY: longint);
  external 'libopenblas.dll';

procedure cblas_sgemm(Order: longint; TransA: longint; TransB: longint;
  M: longint; N: longint; K: longint; alpha: single; A: TSingleVector;
  lda: longint; B: TSingleVector; ldb: longint; beta: single;
  C: TSingleVector; ldc: longint);
  external LIB_NAME;

function LAPACKE_sgesvd(MatrixLayout: longint; JOBU, JOBVT: char;
  M, N: longint; A: TSingleVector; LDA: longint; S, U: TSingleVector;
  LDU: longint; VT: TSingleVector; LDVT: longint; Superb: TSingleVector): longint;
  external LIB_NAME;

implementation

function Add_BLAS(A, B: TMultiArray): TMultiArray;
begin
  Result := AllocateMultiArray(A.Size,
                               False    // do not allocate the memory yet
                               ).Reshape(A.Shape);
  Result.Data := nil;
  Result.Data := CopyVector(B.GetVirtualData);
  cblas_saxpy(A.Size, 1, A.GetVirtualData, 1, Result.Data, 1);
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
    A.GetVirtualData, B.Shape[0],       // A, B.shape[0]
    B.GetVirtualData, B.Shape[1],       // B, B.shape[1]
    1,                                  // beta
    Result.Data, B.Shape[1]             // C, B.shape[1]
    );
end;

end.

